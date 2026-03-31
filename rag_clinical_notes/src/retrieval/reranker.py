"""Cross-encoder re-ranking for retrieved clinical note chunks."""
from __future__ import annotations

import functools
from typing import List


@functools.lru_cache(maxsize=2)
def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Load and cache a CrossEncoder model."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: List[dict],
    reranker,
    top_k: int = 5,
) -> List[dict]:
    """Score all candidates with the cross-encoder and return top_k sorted
    by descending relevance score.

    Each candidate dict gains a "rerank_score" float key.
    Input candidates are the dicts returned by query_collection().
    """
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    scored = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        scored.append(item)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]
