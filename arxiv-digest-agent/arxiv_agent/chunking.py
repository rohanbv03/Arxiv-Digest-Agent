"""
Chunking for the RAG store.

`chunk_pages` is the primary path: it windows over the paper's text while
tracking which page(s) and section each window spans, so every chunk in
the vector store carries {page_start, page_end, section} metadata -- which
is what lets QA answers cite "page 7, Experiments" instead of an opaque
chunk index.

`chunk_text` is kept as a fallback for the no-page-structure case (e.g.
the abstract-only degraded path when PDF parsing fails entirely).

Why word-window instead of a fully semantic chunker: for QA-over-one-paper,
fixed overlapping windows are robust to messy PDF extraction (headers can
be mis-detected, columns can interleave), and the overlap means an
answer-bearing sentence straddling a chunk boundary still appears intact
in at least one chunk. See README "Design Decisions" for the tradeoff.
"""
from typing import List
from . import config
from .state import Chunk


def chunk_text(
    text: str,
    chunk_size_words: int = config.CHUNK_SIZE_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS,
    section: str = "",
) -> List[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks: List[Chunk] = []
    step = max(chunk_size_words - overlap_words, 1)
    idx = 0
    chunk_id = 0
    while idx < len(words):
        window = words[idx : idx + chunk_size_words]
        chunk_str = " ".join(window)
        if chunk_str.strip():
            chunks.append(
                Chunk(chunk_index=chunk_id, text=chunk_str, page_start=1, page_end=1, section=section)
            )
            chunk_id += 1
        idx += step
    return chunks


def chunk_pages(
    pages: List[str],
    page_sections: List[str],
    chunk_size_words: int = config.CHUNK_SIZE_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS,
) -> List[Chunk]:
    """
    Flattens (word, 1-indexed page number, section) triples across all
    pages, then applies the same sliding-window logic as `chunk_text` --
    but each resulting chunk records the page range and section it drew
    from, taken from the first word in its window.
    """
    tagged_words = []  # list of (word, page_num, section)
    for page_idx, page_text in enumerate(pages):
        page_num = page_idx + 1
        section = page_sections[page_idx] if page_idx < len(page_sections) else ""
        for w in page_text.split():
            tagged_words.append((w, page_num, section))

    if not tagged_words:
        return []

    chunks: List[Chunk] = []
    step = max(chunk_size_words - overlap_words, 1)
    idx = 0
    chunk_id = 0
    n = len(tagged_words)
    while idx < n:
        window = tagged_words[idx : idx + chunk_size_words]
        text = " ".join(w for w, _, _ in window)
        if text.strip():
            pages_in_window = [p for _, p, _ in window]
            chunks.append(
                Chunk(
                    chunk_index=chunk_id,
                    text=text,
                    page_start=min(pages_in_window),
                    page_end=max(pages_in_window),
                    section=window[0][2] or "unknown",
                )
            )
            chunk_id += 1
        idx += step
    return chunks
