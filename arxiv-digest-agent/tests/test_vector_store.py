import tempfile

from arxiv_agent.vector_store import VectorStore, sanitize_collection_name


def test_sanitize_collection_name():
    assert sanitize_collection_name("2401.12345") == "paper_2401_12345"
    assert sanitize_collection_name("hep-th/9901001") == "paper_hep-th_9901001"


def test_exists_false_for_unknown_collection():
    with tempfile.TemporaryDirectory() as d:
        store = VectorStore(persist_dir=d)
        assert store.exists("paper_nonexistent") is False
        assert store.count("paper_nonexistent") == 0


def test_add_and_query_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        store = VectorStore(persist_dir=d)
        chunks = [
            {"chunk_index": 0, "text": "hello world", "page_start": 1, "page_end": 1, "section": "intro"},
            {"chunk_index": 1, "text": "goodbye world", "page_start": 2, "page_end": 2, "section": "results"},
        ]
        # simple 2-d embeddings so cosine similarity is easy to reason about
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        store.add_chunks("paper_test", chunks, embeddings, arxiv_id="test.0001")

        assert store.exists("paper_test") is True
        assert store.count("paper_test") == 2

        results = store.query("paper_test", [1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_index"] == 0
        assert results[0]["page_start"] == 1
        assert results[0]["section"] == "intro"


def test_query_on_empty_collection_returns_empty_list():
    with tempfile.TemporaryDirectory() as d:
        store = VectorStore(persist_dir=d)
        assert store.query("paper_empty", [1.0, 0.0], top_k=5) == []
