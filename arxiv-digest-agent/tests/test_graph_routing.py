"""
Graph routing tests -- exercise the conditional edges without hitting any
real API, by monkeypatching the arxiv_client functions the node calls.
"""
import os
os.environ.setdefault("GOOGLE_API_KEY", "dummy-key-for-tests")

from arxiv_agent import graph as graph_module
from arxiv_agent.arxiv_client import ArxivRetrievalError


def test_route_after_retrieval_end_on_error():
    assert graph_module._route_after_retrieval({"error": "boom"}) == "END"


def test_route_after_retrieval_continues_on_success():
    assert graph_module._route_after_retrieval({"error": None}) == "selection_ranking"


def test_node_arxiv_retrieval_zero_results(monkeypatch):
    monkeypatch.setattr(graph_module.arxiv_client, "search_arxiv", lambda q, max_results: [])
    state = {"intent": "topic", "search_query": "nonexistent topic xyz", "user_input": "nonexistent topic xyz"}
    result = graph_module.node_arxiv_retrieval(state)
    assert result["error_kind"] == "no_results"
    assert result["candidates"] == []


def test_node_arxiv_retrieval_api_failure(monkeypatch):
    def boom(q, max_results):
        raise ArxivRetrievalError("network down")

    monkeypatch.setattr(graph_module.arxiv_client, "search_arxiv", boom)
    state = {"intent": "topic", "search_query": "anything", "user_input": "anything"}
    result = graph_module.node_arxiv_retrieval(state)
    assert result["error_kind"] == "api_failure"
    assert "network down" in result["error"]


def test_node_arxiv_retrieval_success(monkeypatch):
    fake_paper = {"arxiv_id": "2401.00001", "title": "Fake Paper"}
    monkeypatch.setattr(graph_module.arxiv_client, "search_arxiv", lambda q, max_results: [fake_paper])
    state = {"intent": "topic", "search_query": "anything", "user_input": "anything"}
    result = graph_module.node_arxiv_retrieval(state)
    assert result["error"] is None
    assert result["candidates"] == [fake_paper]


def test_pipeline_graph_compiles():
    g = graph_module.build_pipeline_graph()
    nodes = set(g.get_graph().nodes.keys())
    assert {"query_understanding", "arxiv_retrieval", "selection_ranking",
            "fetch_and_parse", "chunk_and_embed", "summarize"}.issubset(nodes)


def test_qa_graph_compiles():
    g = graph_module.build_qa_graph()
    nodes = set(g.get_graph().nodes.keys())
    assert "qa_answer" in nodes
