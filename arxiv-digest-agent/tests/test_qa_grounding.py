"""
QA grounding tests using fake LLM/store doubles -- no network calls.
"""
from arxiv_agent import qa as qa_module


class FakeStoreFar:
    def query(self, collection_name, query_embedding, top_k):
        return [{"text": "irrelevant", "distance": 5.0, "chunk_index": 0, "page_start": 1, "page_end": 1, "section": "intro"}]


class FakeStoreEmpty:
    def query(self, collection_name, query_embedding, top_k):
        return []


class FakeStoreClose:
    def query(self, collection_name, query_embedding, top_k):
        return [{"text": "the model achieves 90% accuracy", "distance": 0.05, "chunk_index": 3, "page_start": 7, "page_end": 7, "section": "results"}]


class FakeLLMSupported:
    def embed_query(self, text):
        return [0.1, 0.2]

    def generate_json(self, prompt, temperature=0.1):
        return {"supported": True, "answer": "90% accuracy.", "used_chunk_indices": [3]}


class FakeLLMUnsupported:
    def embed_query(self, text):
        return [0.1, 0.2]

    def generate_json(self, prompt, temperature=0.1):
        return {"supported": False, "answer": "Not discussed in the retrieved context.", "used_chunk_indices": []}


def test_far_distance_never_calls_llm():
    # If the LLM's generate_json were called, it would raise (no method on this double).
    class ExplodingLLM:
        def embed_query(self, text):
            return [0.0, 0.0]

    answer, grounded, sources = qa_module.answer_question(
        "irrelevant question", "col", ExplodingLLM(), FakeStoreFar()
    )
    assert grounded is False
    assert sources == []
    assert "won't guess" in answer or "doesn't appear" in answer


def test_empty_retrieval_returns_not_covered():
    answer, grounded, sources = qa_module.answer_question(
        "anything", "col", FakeLLMSupported(), FakeStoreEmpty()
    )
    assert grounded is False
    assert sources == []


def test_close_distance_and_supported_verdict_returns_grounded_answer():
    answer, grounded, sources = qa_module.answer_question(
        "what accuracy do they report?", "col", FakeLLMSupported(), FakeStoreClose()
    )
    assert grounded is True
    assert "90%" in answer
    assert len(sources) == 1
    assert sources[0]["chunk_index"] == 3


def test_close_distance_but_unsupported_verdict_is_not_grounded():
    answer, grounded, sources = qa_module.answer_question(
        "what accuracy do they report?", "col", FakeLLMUnsupported(), FakeStoreClose()
    )
    assert grounded is False
    assert sources == []


def test_format_sources_empty():
    assert qa_module.format_sources([]) == ""


def test_format_sources_single_page():
    sources = [{"page_start": 7, "page_end": 7, "section": "results"}]
    out = qa_module.format_sources(sources)
    assert "p.7" in out
    assert "results" in out
