"""
ai/retriever.py
---------------
Wraps Vespa query.  Sends the BGE-M3 embeddings and gets back
top-K documents with two-phase hybrid ranking (dense+lexical → ColBERT).
"""

from __future__ import annotations
import os
from vespa.application import Vespa

_vespa: Vespa | None = None


def get_vespa() -> Vespa:
    global _vespa
    if _vespa is None:
        _vespa = Vespa(
            url=os.environ["VESPA_URL"],
            cert=os.environ["VESPA_CERT_PATH"],
            key=os.environ["VESPA_KEY_PATH"],
        )
    return _vespa


def retrieve(
    query_text: str,
    embeddings: dict,
    top_k: int = 20,
) -> list[dict]:
    """
    Run hybrid ANN + BM25 query against Vespa with two-phase ranking.

    Returns list of dicts: {id, text, relevance}
    """
    app = get_vespa()

    response = app.query(
        yql="""
            select id, text from nepali_docs where
            userQuery() or
            ({targetHits: %(target_hits)s}nearestNeighbor(dense_rep, q_dense));
        """ % {"target_hits": top_k},
        ranking="m3hybrid",
        hits=top_k,
        body={
            "input.query(q_dense)":       embeddings["dense_vecs"],
            "input.query(q_lexical)":     embeddings["lexical_weights"],
            "input.query(q_colbert)":     embeddings["colbert_vecs"],
            "input.query(q_len_colbert)": embeddings["q_len_colbert"],
            "timeout": "30s",
        },
        query=query_text,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Vespa query failed [{response.status_code}]: {response.get_json()}"
        )

    results = []
    for hit in response.hits:
        results.append(
            {
                "id":        hit["fields"].get("id", ""),
                "text":      hit["fields"].get("text", ""),
                "relevance": hit.get("relevance", 0.0),
            }
        )
    return results
