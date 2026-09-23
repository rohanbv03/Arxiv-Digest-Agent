from arxiv_agent import chunking


def test_chunk_text_basic_windowing():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunking.chunk_text(text, chunk_size_words=100, overlap_words=20)
    assert len(chunks) > 5
    assert chunks[0]["text"].split()[0] == "word0"
    # overlap: second chunk should start before the first chunk ends
    assert chunks[1]["text"].split()[0] == f"word{100-20}"


def test_chunk_text_empty_input():
    assert chunking.chunk_text("") == []


def test_chunk_pages_tracks_page_and_section():
    pages = [
        " ".join(f"p1w{i}" for i in range(50)),
        " ".join(f"p2w{i}" for i in range(50)),
    ]
    page_sections = ["introduction", "results"]
    chunks = chunking.chunk_pages(pages, page_sections, chunk_size_words=60, overlap_words=10)
    assert len(chunks) >= 1
    # first chunk should start on page 1
    assert chunks[0]["page_start"] == 1
    # a chunk spanning both pages should have page_end == 2
    assert any(c["page_end"] == 2 for c in chunks)
    # section labels should appear
    sections_seen = {c["section"] for c in chunks}
    assert "introduction" in sections_seen or "results" in sections_seen


def test_chunk_pages_empty_input():
    assert chunking.chunk_pages([], []) == []
