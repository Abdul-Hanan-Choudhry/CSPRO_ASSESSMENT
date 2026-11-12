"""Score fusion for hybrid retrieval. Pure functions over (id -> score) maps.

Both methods consume the same dense and sparse candidate lists, so they can be compared
like-for-like. That comparison is how the incident's score-scale bug is demonstrated.
"""

from collections.abc import Mapping
from typing import Literal

FusionMethod = Literal["rrf", "legacy_naive"]


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank), rank starting at 1.

    Uses ranks only, so retrievers with incomparable score scales contribute equally.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def legacy_naive(
    dense: Mapping[str, float], sparse: Mapping[str, float], bm25_weight: float
) -> dict[str, float]:
    """The pre-incident formula: weighted sum of RAW cosine (0..1) and RAW BM25 (unbounded).

    Kept to reproduce the bug: BM25 values of 5-20 swamp cosine differences of ~0.1, so
    keyword-dense documents win regardless of meaning. Do not use in production.
    """
    ids = dense.keys() | sparse.keys()
    return {
        i: (1 - bm25_weight) * dense.get(i, 0.0) + bm25_weight * sparse.get(i, 0.0) for i in ids
    }


def fuse(
    method: FusionMethod,
    dense: Mapping[str, float],
    sparse: Mapping[str, float],
    *,
    rrf_k: int,
    bm25_weight: float,
) -> list[tuple[str, float]]:
    """Return (id, fused_score) sorted best-first; ties broken by id for determinism."""
    if method == "rrf":
        scores = rrf([_ranked(dense), _ranked(sparse)], k=rrf_k)
    else:
        scores = legacy_naive(dense, sparse, bm25_weight)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _ranked(scores: Mapping[str, float]) -> list[str]:
    return [i for i, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
