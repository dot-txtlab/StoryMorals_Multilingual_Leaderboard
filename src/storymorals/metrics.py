"""Similarity-based evaluation — replicates Wu & Piper H3 (Fig 3) and H4 (Fig 4).

H3  "within-language alignment" (Fig 3):
    For each passage (same story, same language) compare cosine similarity of
    human-human (HH) moral pairs vs human-model (HM) pairs. A model that meets
    or exceeds the HH baseline sits inside the human band of variance => good.

H4  "cross-language diversity" (Fig 4):
    For each story, across *different* languages, compare HH cross-lingual pairs
    vs model-model (MM) cross-lingual pairs. Models with MM similarity well
    ABOVE the HH baseline are flattening cultural variation => bad. Low / near
    the human baseline => the model reproduces human cross-cultural diversity.

For each model we report the raw mean gap and a mixed-effects estimate with
random intercepts for story, language(-pair), and embedding model (with an OLS
cluster-robust fallback if the mixed model fails to converge).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------
def _pairs_within(long_df: pd.DataFrame):
    """Return (hh, hm) DataFrames of within-passage pairs (cols: i, j, story, language[, model])."""
    hh_rows, hm_rows = [], []
    for _, grp in long_df.groupby("passage_id"):
        story = grp["country_of_origin"].iloc[0]
        lang = grp["language"].iloc[0]
        humans = grp[grp.is_human]
        hidx = humans["row_id"].tolist()
        for a, b in itertools.combinations(hidx, 2):
            hh_rows.append((a, b, story, lang))
        for src, mgrp in grp[~grp.is_human].groupby("source"):
            for mi in mgrp["row_id"].tolist():
                for hi in hidx:
                    hm_rows.append((hi, mi, story, lang, src))
    hh = pd.DataFrame(hh_rows, columns=["i", "j", "story", "language"])
    hm = pd.DataFrame(hm_rows, columns=["i", "j", "story", "language", "model"])
    return hh, hm


def _pairs_cross(long_df: pd.DataFrame):
    """Return (hh, mm) DataFrames of cross-language, same-story pairs."""
    hh_rows, mm_rows = [], []
    for story, grp in long_df.groupby("country_of_origin"):
        humans = grp[grp.is_human]
        hrec = list(zip(humans["row_id"], humans["language"]))
        for (a, la), (b, lb) in itertools.combinations(hrec, 2):
            if la != lb:
                hh_rows.append((a, b, story, _lp(la, lb)))
        for src, mgrp in grp[~grp.is_human].groupby("source"):
            mrec = list(zip(mgrp["row_id"], mgrp["language"]))
            for (a, la), (b, lb) in itertools.combinations(mrec, 2):
                if la != lb:
                    mm_rows.append((a, b, story, _lp(la, lb), src))
    hh = pd.DataFrame(hh_rows, columns=["i", "j", "story", "lang_pair"])
    mm = pd.DataFrame(mm_rows, columns=["i", "j", "story", "lang_pair", "model"])
    return hh, mm


def _lp(a: str, b: str) -> str:
    return "_".join(sorted((a, b)))


# --------------------------------------------------------------------------
# Similarity over embedders
# --------------------------------------------------------------------------
def _cos(emb: np.ndarray, i, j) -> np.ndarray:
    # embeddings are L2-normalized -> cosine == dot product
    return np.sum(emb[i] * emb[j], axis=1)


def _attach_sims(pairs: pd.DataFrame, emb: dict[str, np.ndarray],
                 wc: np.ndarray) -> pd.DataFrame:
    """Explode one row per (pair, embedder) with cosine sim + word counts."""
    if pairs.empty:
        return pairs.assign(embedder=[], sim=[], wc_i=[], wc_j=[])
    i = pairs["i"].to_numpy()
    j = pairs["j"].to_numpy()
    frames = []
    for name, mat in emb.items():
        f = pairs.copy()
        f["embedder"] = name
        f["sim"] = _cos(mat, i, j)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["wc_i"] = wc[out["i"].to_numpy()]
    out["wc_j"] = wc[out["j"].to_numpy()]
    return out


# --------------------------------------------------------------------------
# Regression (mixed-effects with crossed random intercepts; OLS fallback)
# --------------------------------------------------------------------------
def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def _fit_simple(df: pd.DataFrame) -> tuple[float, float | None]:
    """Per-cell advantage (coef on is_model) + p, OLS clustered by story.

    Used for the per-language H3 breakdown (Fig 9), where language is fixed so
    we control only for word count and cluster on story for the asterisks.
    """
    df = df.copy()
    df["wc_i_z"] = _zscore(df["wc_i"].astype(float))
    df["wc_j_z"] = _zscore(df["wc_j"].astype(float))
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if df["story"].nunique() > 1:
                res = smf.ols("sim ~ is_model + wc_i_z + wc_j_z", df).fit(
                    cov_type="cluster", cov_kwds={"groups": df["story"]})
            else:
                res = smf.ols("sim ~ is_model + wc_i_z + wc_j_z", df).fit(cov_type="HC3")
            return float(res.params["is_model"]), float(res.pvalues["is_model"])
        except Exception:
            adv = (df.loc[df.is_model == 1, "sim"].mean()
                   - df.loc[df.is_model == 0, "sim"].mean())
            return float(adv), None


def _fit_gap(df: pd.DataFrame, re_cols: list[str]) -> dict:
    """Fit sim ~ is_model + wc + crossed RE; return coef/ci/p for is_model."""
    df = df.copy()
    df["wc_i_z"] = _zscore(df["wc_i"].astype(float))
    df["wc_j_z"] = _zscore(df["wc_j"].astype(float))
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df["_grp"] = 1
            vc = {c: f"0 + C({c})" for c in re_cols if df[c].nunique() > 1}
            res = smf.mixedlm("sim ~ is_model + wc_i_z + wc_j_z", df,
                              groups=df["_grp"], vc_formula=vc).fit(reml=False)
            ci = res.conf_int().loc["is_model"]
            return {"coef": float(res.params["is_model"]),
                    "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                    "p": float(res.pvalues["is_model"]), "method": "mixedlm"}
        except Exception:
            # Fallback: OLS with cluster-robust SE by story.
            res = smf.ols("sim ~ is_model + wc_i_z + wc_j_z", df).fit(
                cov_type="cluster", cov_kwds={"groups": df["story"]})
            ci = res.conf_int().loc["is_model"]
            return {"coef": float(res.params["is_model"]),
                    "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                    "p": float(res.pvalues["is_model"]), "method": "ols_clustered"}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def evaluate(long_df: pd.DataFrame, emb: dict[str, np.ndarray]) -> dict:
    """Return per-model H3/H4 results + the human baselines."""
    wc = long_df.sort_values("row_id")["word_count"].to_numpy()

    hh_w, hm_w = _pairs_within(long_df)
    hh_c, mm_c = _pairs_cross(long_df)

    hh_w_s = _attach_sims(hh_w, emb, wc)
    hm_w_s = _attach_sims(hm_w, emb, wc)
    hh_c_s = _attach_sims(hh_c, emb, wc)
    mm_c_s = _attach_sims(mm_c, emb, wc)

    base_within = float(hh_w_s["sim"].mean())
    base_cross = float(hh_c_s["sim"].mean())
    base_within_sd = float(hh_w_s["sim"].std(ddof=1))
    base_cross_sd = float(hh_c_s["sim"].std(ddof=1))

    models = sorted(set(hm_w_s["model"]).union(mm_c_s["model"]))
    results = {}
    for m in models:
        # H3 within-language alignment
        hm = hm_w_s[hm_w_s["model"] == m]
        fit3 = None
        by_language = {}
        if not hm.empty:
            d3 = pd.concat([hh_w_s.assign(is_model=0), hm.assign(is_model=1)],
                           ignore_index=True)
            fit3 = _fit_gap(d3, ["story", "language", "embedder"])
            # Per-language H3 breakdown (Fig 9): model advantage over humans.
            for lang in sorted(hm["language"].unique()):
                hm_l = hm[hm["language"] == lang]
                hh_l = hh_w_s[hh_w_s["language"] == lang]
                d_l = pd.concat([hh_l.assign(is_model=0), hm_l.assign(is_model=1)],
                                ignore_index=True)
                adv, p = _fit_simple(d_l)
                by_language[lang] = {
                    "advantage": round(adv, 4),
                    "p": p,
                    "n": int(len(hm_l) / max(1, len(emb))),
                }

        # H4 cross-language diversity
        mm = mm_c_s[mm_c_s["model"] == m]
        fit4 = None
        if not mm.empty:
            d4 = pd.concat([hh_c_s.assign(is_model=0), mm.assign(is_model=1)],
                           ignore_index=True)
            d4 = d4.rename(columns={"lang_pair": "language"})  # RE name reuse
            fit4 = _fit_gap(d4, ["story", "language", "embedder"])

        hm_mean = float(hm["sim"].mean()) if not hm.empty else float("nan")
        mm_mean = float(mm["sim"].mean()) if not mm.empty else float("nan")
        results[m] = {
            "within": {
                "hm_mean": hm_mean,
                "hh_baseline": base_within,
                "gap_raw": hm_mean - base_within,
                "n_pairs": int(len(hm) / max(1, len(emb))),
                **(fit3 or {}),
            },
            "cross": {
                "mm_mean": mm_mean,
                "hh_baseline": base_cross,
                "gap_raw": mm_mean - base_cross,
                "n_pairs": int(len(mm) / max(1, len(emb))),
                **(fit4 or {}),
            },
            "by_language": by_language,
        }
    return {
        "baselines": {"within": base_within, "cross": base_cross,
                      "within_sd": base_within_sd, "cross_sd": base_cross_sd},
        "n_embedders": len(emb),
        "embedders": list(emb.keys()),
        "models": results,
    }
