"""
ai/embedder.py
--------------
Singleton BGE-M3 embedder.  Loaded once at startup; reused for every query.
"""

from __future__ import annotations
import torch
from FlagEmbedding import BGEM3FlagModel

_model: BGEM3FlagModel | None = None


def get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=(device == "cuda"))
    return _model


def encode_query(query: str) -> dict:
    """
    Returns dict with keys:
        dense_vecs      : list[float]  (1024-d)
        lexical_weights : dict[str, float]
        colbert_vecs    : list[list[float]]   (num_tokens × 1024)
        q_len_colbert   : float
    """
    model = get_model()
    out = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    colbert = out["colbert_vecs"][0]  # ndarray (T, 1024)
    return {
        "dense_vecs": out["dense_vecs"][0].tolist(),
        "lexical_weights": {str(k): float(v) for k, v in out["lexical_weights"][0].items()},
        "colbert_vecs": {str(i): vec.tolist() for i, vec in enumerate(colbert)},
        "q_len_colbert": float(len(colbert)),
    }
