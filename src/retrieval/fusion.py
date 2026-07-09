"""Reciprocal Rank Fusion — combines dense and sparse rankings into one list.

RRF is rank-based, not score-based, so it merges two rankers whose raw scores
aren't comparable (cosine similarity vs. BM25) without normalization headaches."""
from __future__ import annotations


def reciprocal_rank_fusion(
    dense_ranking: list[str],
    sparse_ranking: list[str],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Each list is chunk_ids in rank order (best first). Returns fused (id, score)
    pairs sorted best-first. `k` is the standard RRF damping constant."""
    scores: dict[str, float] = {}

    for rank, chunk_id in enumerate(dense_ranking):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + dense_weight * (1.0 / (k + rank + 1))
    for rank, chunk_id in enumerate(sparse_ranking):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + sparse_weight * (1.0 / (k + rank + 1))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
