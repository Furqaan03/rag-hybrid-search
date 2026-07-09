"""Hybrid index: dense (embeddings) + sparse (BM25) over the same chunks, kept in sync.

Embeddings are pluggable so retrieval logic (fusion, BM25, dedup) tests offline."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.chunking import Chunk
from src.retrieval.fusion import reciprocal_rank_fusion


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class HybridIndex:
    embedder: object  # callable str -> np.ndarray
    chunks: list[Chunk] = field(default_factory=list)
    _ids: list[str] = field(default_factory=list)
    _embeddings: list[np.ndarray] = field(default_factory=list)
    _bm25: BM25Okapi | None = None
    dedup_threshold: float = 0.95

    def _chunk_id(self, chunk: Chunk) -> str:
        return f"{chunk.source}#{chunk.chunk_index}"

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Adds chunks, skipping near-duplicates (cosine > dedup_threshold)."""
        added = 0
        for chunk in chunks:
            vec = self.embedder(chunk.text)
            if self._is_duplicate(vec):
                continue
            self.chunks.append(chunk)
            self._ids.append(self._chunk_id(chunk))
            self._embeddings.append(vec)
            added += 1
        self._rebuild_bm25()
        return added

    def _is_duplicate(self, vec: np.ndarray) -> bool:
        for existing in self._embeddings:
            denom = np.linalg.norm(vec) * np.linalg.norm(existing)
            if denom and float(np.dot(vec, existing) / denom) > self.dedup_threshold:
                return True
        return False

    def _rebuild_bm25(self) -> None:
        if self.chunks:
            self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

    def dense_search(self, query: str, k: int = 10) -> list[str]:
        if not self._embeddings:
            return []
        qvec = self.embedder(query)
        sims = []
        for cid, vec in zip(self._ids, self._embeddings):
            denom = np.linalg.norm(qvec) * np.linalg.norm(vec)
            sims.append((cid, float(np.dot(qvec, vec) / denom) if denom else 0.0))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sims[:k]]

    def sparse_search(self, query: str, k: int = 10) -> list[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:k]]

    def hybrid_search(self, query: str, k: int = 10, dense_weight: float = 0.7, sparse_weight: float = 0.3) -> list[str]:
        dense = self.dense_search(query, k)
        sparse = self.sparse_search(query, k)
        fused = reciprocal_rank_fusion(dense, sparse, dense_weight, sparse_weight)
        return [cid for cid, _ in fused[:k]]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        try:
            return self.chunks[self._ids.index(chunk_id)]
        except ValueError:
            return None
