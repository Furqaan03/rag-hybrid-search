"""Three switchable chunking strategies. Which one wins is measured, not assumed."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    section_heading: str = ""
    strategy: str = ""
    char_count: int = field(default=0)

    def __post_init__(self):
        self.char_count = len(self.text)


def fixed_size_chunks(text: str, source: str, size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Baseline: fixed character windows with overlap."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + size
        chunks.append(Chunk(text=text[start:end], source=source, chunk_index=idx, strategy="fixed"))
        idx += 1
        start += size - overlap
    return chunks


def section_chunks(text: str, source: str) -> list[Chunk]:
    """Structure-aware: split on markdown headers, keeping the heading with its body."""
    parts = re.split(r"(^#{1,6}\s+.*$)", text, flags=re.MULTILINE)
    chunks = []
    idx = 0
    heading = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{1,6}\s+", part):
            heading = part.lstrip("# ").strip()
        else:
            chunks.append(Chunk(text=part, source=source, chunk_index=idx, section_heading=heading, strategy="section"))
            idx += 1
    return chunks or fixed_size_chunks(text, source)


def semantic_chunks(text: str, source: str, embedder=None, threshold: float = 0.5) -> list[Chunk]:
    """Topic-boundary splitting: group adjacent sentences while embedding similarity
    stays high; start a new chunk when the topic shifts. Falls back to sentence
    grouping if no embedder is supplied (offline)."""
    import numpy as np

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []
    if embedder is None:
        # Offline fallback: group every ~4 sentences.
        chunks = []
        for i in range(0, len(sentences), 4):
            chunks.append(Chunk(text=" ".join(sentences[i:i + 4]), source=source, chunk_index=i // 4, strategy="semantic"))
        return chunks

    vecs = [embedder(s) for s in sentences]
    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(vecs[i], vecs[i - 1]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i - 1]) + 1e-9))
        if sim >= threshold:
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])
    return [Chunk(text=" ".join(g), source=source, chunk_index=i, strategy="semantic") for i, g in enumerate(groups)]


STRATEGIES = {"fixed": fixed_size_chunks, "section": section_chunks, "semantic": semantic_chunks}


def chunk_document(text: str, source: str, strategy: str = "section") -> list[Chunk]:
    return STRATEGIES[strategy](text, source)
