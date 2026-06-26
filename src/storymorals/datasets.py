"""Build the unified long-format morals table (human + model) used by eval.

Columns:
    passage_id          e.g. "italy_ar"
    country_of_origin   story origin (e.g. "italy")
    language            passage language code (e.g. "ar")
    source              "human:1/2/3"  or a model id  or "gpt-4o (paper)"
    is_human            bool
    moral               moral text in the passage language
    moral_english       English translation if available (humans + paper GPT), else NaN
    word_count          word count of `moral`
"""
from __future__ import annotations

import pandas as pd

from .config import HUMAN_MORALS_CSV
from .generate import OUT_CSV as MODEL_MORALS_CSV

# The paper's own GPT-4o morals ship inside human_morals.csv (annotator == "gpt").
# We surface them as a reference model so the leaderboard has a baseline point
# and the evaluation path is runnable before any new generation.
PAPER_GPT_SOURCE = "gpt-4o (paper)"


def _word_count(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.split().str.len()


def load_long(include_paper_gpt: bool = True) -> pd.DataFrame:
    h = pd.read_csv(HUMAN_MORALS_CSV)
    h = h.rename(columns={"country.of.origin": "country_of_origin"})
    ann = h["annotator"].astype(str)

    human = h[ann.isin(["1", "2", "3"])].copy()
    human["source"] = "human:" + ann[ann.isin(["1", "2", "3"])]
    human["is_human"] = True

    frames = [human[["passage_id", "country_of_origin", "language",
                     "source", "is_human", "moral", "moral_english"]]]

    if include_paper_gpt and (ann == "gpt").any():
        g = h[ann == "gpt"].copy()
        g["source"] = PAPER_GPT_SOURCE
        g["is_human"] = False
        frames.append(g[["passage_id", "country_of_origin", "language",
                         "source", "is_human", "moral", "moral_english"]])

    if MODEL_MORALS_CSV.exists():
        m = pd.read_csv(MODEL_MORALS_CSV)
        m = m[~m["moral"].astype(str).str.startswith("MODEL_ERROR")].copy()
        m["is_human"] = False
        m["moral_english"] = pd.NA
        frames.append(m[["passage_id", "country_of_origin", "language",
                         "source", "is_human", "moral", "moral_english"]])

    out = pd.concat(frames, ignore_index=True)
    out["moral"] = out["moral"].astype(str)
    out["word_count"] = _word_count(out["moral"])
    out = out.reset_index(drop=True)
    out.insert(0, "row_id", out.index)
    return out
