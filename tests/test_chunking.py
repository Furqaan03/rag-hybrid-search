from src.ingestion.chunking import chunk_document, fixed_size_chunks, section_chunks


def test_fixed_size_overlap():
    text = "abcdefghij" * 20  # 200 chars
    chunks = fixed_size_chunks(text, "doc", size=80, overlap=20)
    assert len(chunks) > 1
    assert all(c.strategy == "fixed" for c in chunks)
    assert all(c.char_count <= 80 for c in chunks)


def test_section_chunks_split_on_headings():
    text = "# Intro\nHello world.\n\n## Details\nMore text here.\n\n## Config\nSet the port."
    chunks = section_chunks(text, "doc")
    headings = {c.section_heading for c in chunks}
    assert "Intro" in headings
    assert "Config" in headings


def test_chunk_document_dispatch():
    chunks = chunk_document("# A\nbody text", "doc", strategy="section")
    assert chunks[0].strategy == "section"
