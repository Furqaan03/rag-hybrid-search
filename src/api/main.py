"""FastAPI RAG service: ingest, ask (hybrid retrieval + grounded gen), list docs."""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from src.generation.generate import generate_answer
from src.ingestion.chunking import chunk_document
from src.ingestion.embeddings import openai_embedder
from src.retrieval.index import HybridIndex
from src.retrieval.rerank import rerank

load_dotenv()

app = FastAPI(title="RAG with Hybrid Search")
_index: HybridIndex | None = None


def get_index() -> HybridIndex:
    global _index
    if _index is None:
        _index = HybridIndex(embedder=openai_embedder())
    return _index


class IngestRequest(BaseModel):
    source: str
    text: str
    strategy: str = "section"


@app.post("/v1/ingest")
def ingest(req: IngestRequest) -> dict:
    chunks = chunk_document(req.text, req.source, req.strategy)
    added = get_index().add_chunks(chunks)
    return {"source": req.source, "chunks_added": added, "strategy": req.strategy}


class AskRequest(BaseModel):
    question: str
    k: int = 5
    hybrid: bool = True


@app.post("/v1/ask")
def ask(req: AskRequest) -> dict:
    index = get_index()
    if req.hybrid:
        candidate_ids = index.hybrid_search(req.question, k=20)
    else:
        candidate_ids = index.dense_search(req.question, k=20)

    candidates = [index.get_chunk(cid) for cid in candidate_ids]
    candidates = [c for c in candidates if c is not None]
    top_chunks = rerank(req.question, candidates, top_n=req.k)

    # Retrieval confidence proxy: fraction of requested chunks actually found.
    retrieval_confidence = min(1.0, len(top_chunks) / max(1, req.k))
    answer = generate_answer(req.question, top_chunks, retrieval_confidence)

    return {
        "answer": answer.answer,
        "citations": [c.model_dump() for c in answer.citations],
        "confidence": {
            "retrieval": answer.retrieval_confidence,
            "citation_coverage": answer.citation_coverage,
            "composite": answer.composite_confidence,
        },
        "sources": [{"id": f"{c.source}#{c.chunk_index}", "heading": c.section_heading} for c in top_chunks],
    }


@app.get("/v1/documents")
def documents() -> dict:
    index = get_index()
    sources = sorted({c.source for c in index.chunks})
    return {"documents": sources, "total_chunks": len(index.chunks)}
