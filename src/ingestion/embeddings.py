"""Pluggable embedder — OpenAI for prod, deterministic hash-based for offline tests."""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np


def openai_embedder(model: str = "text-embedding-3-small") -> Callable[[str], np.ndarray]:
    from openai import OpenAI

    client = OpenAI()

    def embed(text: str) -> np.ndarray:
        return np.array(client.embeddings.create(model=model, input=text).data[0].embedding, dtype=float)

    return embed


def deterministic_fake_embedder(dim: int = 64) -> Callable[[str], np.ndarray]:
    """Bag-of-words hashed into a vector so lexically similar text -> similar vectors.
    Good enough to exercise dense retrieval / dedup offline."""

    def embed(text: str) -> np.ndarray:
        vec = np.zeros(dim)
        for token in text.lower().split():
            h = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
            vec[h % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    return embed
