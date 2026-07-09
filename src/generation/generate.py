"""Grounded generation with inline citations + citation verification + confidence."""
from __future__ import annotations

import json
import re

from openai import OpenAI
from pydantic import BaseModel

from src.ingestion.chunking import Chunk

_SYSTEM = """You answer strictly from the provided numbered context blocks. Cite every
claim with bracketed references like [1], [2] matching the block numbers. If the context
does not contain enough information, say so explicitly — do not invent facts."""


class Citation(BaseModel):
    marker: int
    supported: bool
    verdict_reason: str = ""


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval_confidence: float
    citation_coverage: float
    composite_confidence: float
    insufficient_context: bool


def _build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))


def generate_answer(query: str, chunks: list[Chunk], retrieval_confidence: float, client: OpenAI | None = None) -> GroundedAnswer:
    client = client or OpenAI()

    if retrieval_confidence < 0.3 or not chunks:
        return GroundedAnswer(
            answer="The retrieved context does not contain enough information to answer this question confidently.",
            citations=[], retrieval_confidence=retrieval_confidence, citation_coverage=0.0,
            composite_confidence=retrieval_confidence, insufficient_context=True,
        )

    context = _build_context(chunks)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0,
    )
    answer = resp.choices[0].message.content or ""

    markers = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
    citations = [_verify_citation(m, answer, chunks, client) for m in markers if 1 <= m <= len(chunks)]

    citation_coverage = (sum(1 for c in citations if c.supported) / len(citations)) if citations else 0.0
    composite = round(0.5 * retrieval_confidence + 0.5 * citation_coverage, 3)

    return GroundedAnswer(
        answer=answer, citations=citations, retrieval_confidence=round(retrieval_confidence, 3),
        citation_coverage=round(citation_coverage, 3), composite_confidence=composite, insufficient_context=False,
    )


def _verify_citation(marker: int, answer: str, chunks: list[Chunk], client: OpenAI) -> Citation:
    """Checks whether the cited chunk actually supports the claim it's attached to."""
    chunk = chunks[marker - 1]
    prompt = (
        f"Does this source passage support claims attributed to citation [{marker}] in the answer?\n\n"
        f"Answer: {answer}\n\nSource [{marker}]: {chunk.text[:800]}\n\n"
        'Respond as JSON: {"supported": true/false, "reason": "one sentence"}.'
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    return Citation(marker=marker, supported=bool(parsed.get("supported", False)), verdict_reason=parsed.get("reason", ""))
