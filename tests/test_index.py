from src.ingestion.chunking import Chunk
from src.ingestion.embeddings import deterministic_fake_embedder
from src.retrieval.index import HybridIndex


def _index():
    return HybridIndex(embedder=deterministic_fake_embedder())


def _chunk(text, i):
    return Chunk(text=text, source="doc", chunk_index=i)


def test_sparse_search_finds_exact_keyword():
    idx = _index()
    idx.add_chunks([
        _chunk("The server listens on port 8080 by default", 0),
        _chunk("Cats are wonderful pets and animals", 1),
        _chunk("Database connection uses DATABASE_URL variable", 2),
    ])
    results = idx.sparse_search("port 8080", k=1)
    assert results[0] == "doc#0"


def test_dedup_skips_near_duplicates():
    idx = _index()
    idx.add_chunks([_chunk("identical content here", 0)])
    added = idx.add_chunks([_chunk("identical content here", 1)])
    assert added == 0  # exact duplicate skipped
    assert len(idx.chunks) == 1


def test_hybrid_search_returns_results():
    idx = _index()
    idx.add_chunks([
        _chunk("The API server listens on port 8080", 0),
        _chunk("Set DATABASE_URL for the database", 1),
        _chunk("Enable caching with REDIS_URL", 2),
    ])
    results = idx.hybrid_search("what port does the server use", k=2)
    assert "doc#0" in results


def test_get_chunk_roundtrip():
    idx = _index()
    idx.add_chunks([_chunk("hello", 0)])
    assert idx.get_chunk("doc#0").text == "hello"
    assert idx.get_chunk("doc#99") is None
