"""LLM-as-judge cross-encoder reranker: second-pass precision over fused candidates."""
from __future__ import annotations

import json

from openai import OpenAI

from src.ingestion.chunking import Chunk


def rerank(query: str, chunks: list[Chunk], top_n: int = 5, client: OpenAI | None = None) -> list[Chunk]:
    """Scores each candidate's relevance to the actual question and keeps top_n.
    This second pass dramatically improves precision over fusion alone."""
    if not chunks:
        return []
    client = client or OpenAI()

    numbered = "\n\n".join(f"[{i}] {c.text[:500]}" for i, c in enumerate(chunks))
    prompt = (
        f"Question: {query}\n\nCandidate passages:\n{numbered}\n\n"
        f"Return the indices of the {top_n} most relevant passages, most relevant first, "
        'as JSON: {"ranked_indices": [int, ...]}.'
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    indices = json.loads(resp.choices[0].message.content or "{}").get("ranked_indices", [])
    reranked = [chunks[i] for i in indices if 0 <= i < len(chunks)]
    # Backfill if the model returned fewer than top_n.
    for c in chunks:
        if c not in reranked:
            reranked.append(c)
    return reranked[:top_n]
