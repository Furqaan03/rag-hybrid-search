# RAG Pipeline with Hybrid Search Over Internal Docs

A production-grade Retrieval-Augmented Generation system. It ingests internal
docs, indexes them with both dense vector search and sparse BM25 keyword search,
fuses the two rankings, reranks for precision, and generates grounded answers
with inline citations that are individually verified against their source.

## Why this exists

RAG is the single most-requested skill in AI eng job descriptions, but most
candidates build a toy that embeds one PDF and calls it done. This is the
production version: hybrid retrieval, a real chunking-strategy decision backed
by eval data, citation verification, and graceful "I don't know" handling.

## Architecture

```
src/ingestion/chunking.py     three switchable strategies: fixed+overlap,
                               section-aware, semantic (topic-boundary)
src/ingestion/embeddings.py   pluggable embedder (OpenAI / offline fake)
src/retrieval/index.py        dense + BM25 sparse over the same chunks, with
                               near-duplicate dedup, kept in sync
src/retrieval/fusion.py       Reciprocal Rank Fusion of dense + sparse rankings
src/retrieval/rerank.py       LLM cross-encoder rerank — second-pass precision
src/generation/generate.py    grounded generation + per-citation verification +
                               composite confidence + "I don't know" handling
src/eval/                      golden Q&A set + retrieval metrics (recall@k, MRR)
src/api/main.py               FastAPI: /v1/ingest, /v1/ask, /v1/documents
```

## Design decisions

- **Hybrid beats dense-only for technical docs.** Pure semantic search misses
  exact tokens — function names, config keys, error codes (`DATABASE_URL`,
  `port 8080`). BM25 catches those; dense catches paraphrases. RRF fuses both.
- **RRF is rank-based, not score-based.** Cosine similarity and BM25 scores live
  on incomparable scales; normalizing them is fiddly and brittle. RRF combines by
  rank position, so the two rankers merge cleanly without score normalization.
- **A reranker runs after fusion.** The top ~20 fused candidates go through an
  LLM cross-encoder that scores true relevance to the actual question, keeping the
  top 5. This second pass is where most of the precision gain comes from.
- **Every citation is verified.** After generation, each `[n]` marker is checked
  against the chunk it points at — does that source actually support the claim?
  Unsupported citations are flagged. This is the quality layer most RAG demos skip.
- **"I don't know" is a first-class outcome.** Below a retrieval-confidence
  threshold the system refuses rather than hallucinating, returning what it did and
  didn't find. That's more useful than a confident fabrication and signals maturity.
- **Embedder is injected.** Chunking, BM25, RRF, and dedup are all tested offline
  with a deterministic bag-of-words fake embedder — no API key needed for the suite.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env      # fill in OPENAI_API_KEY
uvicorn src.api.main:app --reload
```

## Example

```bash
curl -X POST localhost:8000/v1/ingest -H "Content-Type: application/json" \
  -d '{"source": "config.md", "text": "# Server\nListens on port 8080...", "strategy": "section"}'

curl -X POST localhost:8000/v1/ask -H "Content-Type: application/json" \
  -d '{"question": "what port does the server use?", "hybrid": true}'
# -> answer with [n] citations, per-citation verification, composite confidence, sources
```

## Tests

```bash
pytest tests/ -v
```

13 tests covering RRF fusion (both-list boost, weighting, id preservation),
chunking (overlap, section splitting, dispatch), the hybrid index (BM25 exact-
keyword match, dedup, hybrid search, roundtrip), and retrieval metrics
(recall@k, MRR, hit rate) — all offline via the deterministic fake embedder.

## Docker

```bash
docker build -t rag-hybrid . && docker run -p 8000:8000 --env-file .env rag-hybrid
```

## Status

Phases 1-4 complete (ingestion+chunking, hybrid retrieval+rerank, grounded
generation+citation verification, eval metrics + golden Q&A). ChromaDB is a
dependency for the persistent-vector-store path; the default index is in-memory
numpy for portability. Phase 5's comparison dashboard is a manual eval run.
