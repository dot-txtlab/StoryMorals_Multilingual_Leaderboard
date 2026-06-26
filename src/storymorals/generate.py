"""Generate model story-morals for every passage (story x language), per model.

Reads passages.csv + the socio-demographic prompt + models.yaml, calls each
enabled model's API on all 196 passages, and writes a long-format CSV:

    output/model_morals.csv
    columns: passage_id, language, country_of_origin, source, moral

`source` is the model id. Human morals live separately in data/human_morals.csv
and are merged at evaluation time.

Resilience:
  * Each call has a timeout + bounded exponential-backoff retry (providers.py).
  * A failed passage is written as "MODEL_ERROR: ..." (never silently dropped).
  * Each passage is written to CSV the moment it completes, so outputs
    accumulate live and a crash loses at most the one call in flight.
  * Re-running RESUMES: finished passages are skipped, and previously FAILED
    passages are retried (their error rows are replaced).
  * A fatal error (bad key / unknown model) short-circuits that model instead
    of hammering all 196 passages.
"""
from __future__ import annotations

import concurrent.futures as cf
import threading

import pandas as pd
from tqdm import tqdm

from . import OUTPUT
from .config import (PASSAGES_CSV, load_languages, load_models,
                     load_prompt_template)
from .providers import FatalProviderError, ProviderError, generate

OUT_CSV = OUTPUT / "model_morals.csv"
_COLS = ["passage_id", "language", "country_of_origin", "source", "moral"]


def _fill_prompt(template: str, lang_code: str, languages: dict, passage: str) -> str:
    lang = languages[lang_code]
    return (template
            .replace("{LANGUAGE}", lang["name"])
            .replace("{COUNTRY}", lang["country"])
            .replace("{PASSAGE}", passage))


def _load_existing() -> pd.DataFrame:
    if OUT_CSV.exists():
        return pd.read_csv(OUT_CSV)
    return pd.DataFrame(columns=_COLS)


def _is_error(s: pd.Series) -> pd.Series:
    return s.astype(str).str.startswith("MODEL_ERROR")


def _append_row(row: dict) -> None:
    """Append one completed passage to the CSV immediately (crash-safe)."""
    header = not OUT_CSV.exists()
    pd.DataFrame([row])[_COLS].to_csv(OUT_CSV, mode="a", header=header, index=False)


def run(only_model: str | None = None, limit: int | None = None,
        story: str | None = None) -> pd.DataFrame:
    OUTPUT.mkdir(exist_ok=True)
    specs, providers, defaults = load_models()
    languages = load_languages()
    template = load_prompt_template()
    passages = pd.read_csv(PASSAGES_CSV)

    if story:
        passages = passages[passages["country.of.origin"] == story]
        if passages.empty:
            raise SystemExit(f"No passages for story/country {story!r}.")
    if limit:
        passages = passages.head(limit)
    n_pass = len(passages)
    want_ids = set(passages["passage_id"])

    existing = _load_existing()
    specs = [s for s in specs if only_model is None or s.id == only_model]
    if not specs:
        raise SystemExit(f"No enabled model matches {only_model!r} in models.yaml.")

    timeout = float(defaults.get("request_timeout", 120))
    concurrency = int(defaults.get("concurrency", 8))
    max_retries = int(defaults.get("max_retries", 5))

    for spec in specs:
        mine = existing[existing.source == spec.id]
        good_ids = set(mine.loc[~_is_error(mine["moral"]), "passage_id"])
        todo_ids = want_ids - good_ids
        if not todo_ids:
            print(f"[skip] {spec.display} ({spec.id}) — all {n_pass} done.")
            continue
        todo = passages[passages["passage_id"].isin(todo_ids)]
        print(f"[run]  {spec.display} ({spec.id}) — {len(todo)} passages "
              f"({len(good_ids)} already done)")

        # Clean base: drop this model's stale rows for the passages we're about
        # to (re)run, then rewrite the CSV once so disk matches before we start
        # appending fresh results row-by-row.
        keep = ~((existing.source == spec.id) & (existing.passage_id.isin(todo_ids)))
        existing = existing[keep].reset_index(drop=True)
        existing.to_csv(OUT_CSV, index=False)

        pcfg = providers[spec.provider]
        aborted = threading.Event()   # set on a fatal error -> short-circuit rest

        def _one(rec):
            if aborted.is_set():
                return _err_row(rec, spec.id, "skipped (model aborted)")
            prompt = _fill_prompt(template, rec["language"], languages,
                                  rec["translated.summary"])
            try:
                moral = generate(spec.provider, pcfg, spec.id, prompt,
                                 max_tokens=spec.max_tokens,
                                 temperature=spec.temperature,
                                 max_retries=max_retries, timeout=timeout)
            except FatalProviderError as exc:
                aborted.set()
                return _err_row(rec, spec.id, str(exc))
            except ProviderError as exc:
                return _err_row(rec, spec.id, str(exc))
            return {"passage_id": rec["passage_id"], "language": rec["language"],
                    "country_of_origin": rec["country.of.origin"],
                    "source": spec.id, "moral": moral}

        records = todo.to_dict("records")
        rows = []
        # ex.map yields in the main thread, so per-result CSV appends are serial
        # and safe even though the calls run concurrently.
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            for row in tqdm(ex.map(_one, records), total=len(records),
                            desc=spec.id, unit="passage"):
                _append_row(row)        # write each passage as soon as it lands
                rows.append(row)

        existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        n_err = sum(_is_error(pd.Series([r["moral"] for r in rows])))
        if aborted.is_set():
            print(f"       ABORTED — fatal error, no more calls for this model. "
                  f"Fix and re-run. ({n_err} error rows)")
        else:
            print(f"       done — {len(rows)} morals ({n_err} errors). Saved {OUT_CSV}")

    return existing


def _err_row(rec, source, msg):
    return {"passage_id": rec["passage_id"], "language": rec["language"],
            "country_of_origin": rec["country.of.origin"], "source": source,
            "moral": f"MODEL_ERROR: {msg}"}


def status() -> None:
    """Print per-model progress and a sample of errors (troubleshooting)."""
    specs, _, _ = load_models()
    n_pass = len(pd.read_csv(PASSAGES_CSV))
    ex = _load_existing()
    print(f"Passages per model: {n_pass}\n" + "-" * 64)
    if ex.empty:
        print("No model_morals.csv yet — run `generate`.")
        return
    for spec in specs:
        mine = ex[ex.source == spec.id]
        if mine.empty:
            print(f"  {spec.display:<24} not started")
            continue
        err = mine[_is_error(mine["moral"])]
        good = len(mine) - len(err)
        missing = n_pass - mine["passage_id"].nunique()
        flag = "OK " if (good == n_pass) else "!! "
        print(f"{flag}{spec.display:<24} done={good:>3}  errors={len(err):>3}  missing={missing:>3}")
        if len(err):
            sample = err["moral"].iloc[0][:90]
            print(f"      e.g. {sample}")
    print("-" * 64)
    print("Re-run `generate` to retry errored/missing passages "
          "(finished ones are skipped).")
