"""Embed morals with multilingual sentence encoders (cached to output/).

The paper uses three encoders (LaBSE, multilingual-MiniLM, English MPNet).
We default to the two *multilingual* ones, which operate on the original-language
moral directly — no translation step required, and results are robust across
encoders (the paper includes embedding model as a random effect). The English
MPNet option is available but needs `moral_english` (present for humans + the
paper's GPT; new model morals would need a translation pass first).

A `tfidf` encoder is provided as a fast, dependency-light DEV option to exercise
the pipeline without downloading models — it is NOT meaningful cross-lingually
and must not be used for published numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import OUTPUT

EMBEDDERS = {
    # Modern multilingual primary (default). ~2.3 GB; downloaded on first use.
    "bge-m3": "BAAI/bge-m3",
    # Strong, well-understood multilingual encoders (paper's choices).
    "labse": "sentence-transformers/LaBSE",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # English-only; needs moral_english translations (paper's third encoder).
    "mpnet-en": "sentence-transformers/all-mpnet-base-v2",
}

# Default: a current SOTA encoder + a classic anchor, averaged via the
# embedding-model random effect (keeps results from hinging on one encoder).
DEFAULT_EMBEDDERS = ["bge-m3", "labse"]

# For apples-to-apples comparison with the published Figs 3 & 4:
#   python scripts/run.py evaluate --embedders labse minilm mpnet-en
PAPER_EMBEDDERS = ["labse", "minilm", "mpnet-en"]

_EMB_DIR = OUTPUT / "embeddings"


def _text_for(name: str, long_df: pd.DataFrame) -> pd.Series:
    if name == "mpnet-en":
        return long_df["moral_english"].fillna(long_df["moral"]).astype(str)
    return long_df["moral"].astype(str)


def _encode_st(model_id: str, texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)


def _encode_tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(min_df=1, max_features=4096)
    mat = vec.fit_transform(texts)
    return normalize(mat.toarray()).astype(np.float32)


def embed_all(long_df: pd.DataFrame, names: list[str] | None = None,
              force: bool = False) -> dict[str, np.ndarray]:
    """Return {embedder_name: (n_rows, dim) array aligned to long_df.row_id order}."""
    names = names or DEFAULT_EMBEDDERS
    _EMB_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    for name in names:
        cache = _EMB_DIR / f"{name}_{len(long_df)}.npy"
        if cache.exists() and not force:
            out[name] = np.load(cache)
            continue
        texts = _text_for(name, long_df).tolist()
        print(f"[embed] {name} on {len(texts)} morals ...")
        if name == "tfidf":
            arr = _encode_tfidf(texts)
        else:
            arr = _encode_st(EMBEDDERS[name], texts)
        arr = np.asarray(arr, dtype=np.float32)
        np.save(cache, arr)
        out[name] = arr
    return out
