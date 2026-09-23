"""
Two separate compiled graphs, deliberately kept apart:

  build_pipeline_graph() -- runs once per paper: query understanding all
  the way through summarization. Ends at END; no interactive input() calls
  live inside a graph node (the earlier version had `input()` inside the
  QA node, which mixes orchestration with a CLI-specific concern). Where
  the pipeline needs a human choice (picking among topic-search
  candidates), that choice is still made by a human, but via a plain
  Python callback the CLI supplies -- not via a hardcoded `input()`
  call baked into the node, so the same graph can run non-interactively
  (e.g. under a "pick top match" callback) or headless in tests.

  build_qa_graph() -- one node, one turn: given `pending_question` and a
  `collection_name` already produced by the pipeline graph, retrieve +
  verify + answer. The CLI invokes this graph once per question in its own
  loop, which is what makes `pending_question` (previously declared but
  unused in state) actually load-bearing.

Routing:
  pipeline: query_understanding -> arxiv_retrieval
              -(no results)-> END (error_kind="no_results")
              -(api failure)-> END (error_kind="api_failure")
              -(ok)-> selection_ranking -> fetch_and_parse -> chunk_and_embed -> summarize -> END
  qa:       qa_answer -> END
"""
import json
import math
import os
from typing import Callable, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from . import config
from .state import AgentState
from . import arxiv_client, pdf_utils, chunking, qa as qa_module
from .arxiv_client import ArxivRetrievalError
from .pdf_utils import PdfParseError
from .llm import GeminiClient
from .vector_store import VectorStore, sanitize_collection_name
from .briefing import build_briefing, briefing_to_markdown

SelectionCallback = Callable[[List[dict]], int]


def _default_selection_callback(scored_candidates: List[dict]) -> int:
    """Non-interactive default: just take the top-ranked candidate."""
    return 0


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------- pipeline nodes --

def node_query_understanding(state: AgentState) -> dict:
    user_input = state["user_input"]
    normalized = arxiv_client.extract_arxiv_id(user_input)
    if normalized:
        return {"intent": "id", "normalized_id": normalized, "search_query": None}
    cleaned = arxiv_client.clean_topic_query(user_input)
    return {"intent": "topic", "normalized_id": None, "search_query": cleaned}


def node_arxiv_retrieval(state: AgentState) -> dict:
    try:
        if state["intent"] == "id":
            candidates = arxiv_client.fetch_by_id(state["normalized_id"])
        else:
            candidates = arxiv_client.search_arxiv(
                state["search_query"] or state["user_input"],
                max_results=config.MAX_ARXIV_CANDIDATES,
            )
    except ArxivRetrievalError as e:
        return {"candidates": [], "error": str(e), "error_kind": "api_failure"}

    if not candidates:
        return {
            "candidates": [],
            "error": (
                "No papers found on arXiv for that query. Try a broader or "
                "differently-worded topic, or double-check the arXiv ID/URL."
            ),
            "error_kind": "no_results",
        }
    return {"candidates": candidates, "error": None, "error_kind": None}


def _route_after_retrieval(state: AgentState) -> str:
    return "END" if state.get("error") else "selection_ranking"


def node_selection_ranking(state: AgentState, llm: GeminiClient, on_select: SelectionCallback) -> dict:
    candidates = state["candidates"]
    if state["intent"] == "id" or len(candidates) == 1:
        return {"selected_paper": candidates[0]}

    query_emb = llm.embed_query(state["search_query"] or state["user_input"])
    scored = []
    for c in candidates:
        abs_emb = llm.embed_query(c.get("abstract", "") or c.get("title", ""))
        scored.append((_cosine_sim(query_emb, abs_emb), c))
    scored.sort(key=lambda x: x[0], reverse=True)

    idx = on_select([{"score": s, "paper": c} for s, c in scored])
    idx = idx if 0 <= idx < len(scored) else 0
    return {"selected_paper": scored[idx][1]}


def node_fetch_and_parse(state: AgentState) -> dict:
    paper = state["selected_paper"]
    try:
        path = pdf_utils.download_pdf(paper["pdf_url"], paper["arxiv_id"])
        pages, page_sections, sections, references, warning = pdf_utils.parse_pdf(path)
        return {
            "pages": pages,
            "page_sections": page_sections,
            "sections": sections,
            "references": references,
            "parse_warning": warning,
            "used_abstract_fallback": False,
        }
    except PdfParseError as e:
        # Graceful degradation: fall back to abstract-only so the pipeline
        # still produces a (clearly-labeled, weaker) briefing instead of
        # dying outright.
        warning = f"PDF fetch/parse failed ({e}); falling back to abstract-only mode."
        print(f"[warning] {warning}")
        abstract = paper.get("abstract", "")
        return {
            "pages": [abstract],
            "page_sections": ["abstract"],
            "sections": {"abstract": abstract},
            "references": [],
            "parse_warning": warning,
            "used_abstract_fallback": True,
        }


def node_chunk_and_embed(state: AgentState, llm: GeminiClient, store: VectorStore) -> dict:
    paper = state["selected_paper"]
    collection_name = sanitize_collection_name(paper["arxiv_id"])

    if store.exists(collection_name):
        # Real index reuse: an earlier run already embedded this exact
        # paper into this Chroma collection on disk, so skip re-parsing
        # cost entirely rather than re-adding (the old code always
        # deleted-then-re-added on every run, which only *looked* like
        # reuse).
        print(f"[info] existing vector index found for {paper['arxiv_id']}; reusing it")
        return {
            "collection_name": collection_name,
            "num_chunks": store.count(collection_name),
            "index_reused": True,
        }

    chunks = chunking.chunk_pages(state["pages"], state["page_sections"])
    if not chunks:
        chunks = chunking.chunk_text(paper.get("abstract", ""), section="abstract")

    embeddings = llm.embed_texts([c["text"] for c in chunks])
    store.add_chunks(collection_name, chunks, embeddings, arxiv_id=paper["arxiv_id"])

    return {"collection_name": collection_name, "num_chunks": len(chunks), "index_reused": False}


def node_summarize(state: AgentState, llm: GeminiClient) -> dict:
    briefing = build_briefing(
        state["selected_paper"], state["sections"], state.get("parse_warning", ""), llm
    )

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    base = state["selected_paper"]["arxiv_id"].replace("/", "_")
    json_path = os.path.join(config.OUTPUT_DIR, f"{base}_briefing.json")
    md_path = os.path.join(config.OUTPUT_DIR, f"{base}_briefing.md")
    with open(json_path, "w") as f:
        json.dump(briefing, f, indent=2)
    with open(md_path, "w") as f:
        f.write(briefing_to_markdown(briefing))

    print("\n" + "=" * 70)
    print(briefing_to_markdown(briefing))
    print("=" * 70)
    print(f"(saved to {json_path} and {md_path})\n")

    return {"briefing": briefing}


def build_pipeline_graph(
    llm: Optional[GeminiClient] = None,
    store: Optional[VectorStore] = None,
    on_select: SelectionCallback = _default_selection_callback,
):
    llm = llm or GeminiClient()
    store = store or VectorStore()

    graph = StateGraph(AgentState)
    graph.add_node("query_understanding", node_query_understanding)
    graph.add_node("arxiv_retrieval", node_arxiv_retrieval)
    graph.add_node("selection_ranking", lambda s: node_selection_ranking(s, llm, on_select))
    graph.add_node("fetch_and_parse", node_fetch_and_parse)
    graph.add_node("chunk_and_embed", lambda s: node_chunk_and_embed(s, llm, store))
    graph.add_node("summarize", lambda s: node_summarize(s, llm))

    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "arxiv_retrieval")
    graph.add_conditional_edges(
        "arxiv_retrieval", _route_after_retrieval, {"END": END, "selection_ranking": "selection_ranking"}
    )
    graph.add_edge("selection_ranking", "fetch_and_parse")
    graph.add_edge("fetch_and_parse", "chunk_and_embed")
    graph.add_edge("chunk_and_embed", "summarize")
    graph.add_edge("summarize", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------- qa graph --

def node_qa_answer(state: AgentState, llm: GeminiClient, store: VectorStore) -> dict:
    question = state["pending_question"]
    answer, grounded, sources = qa_module.answer_question(question, state["collection_name"], llm, store)
    return {"qa_answer": answer, "qa_grounded": grounded, "qa_sources": sources}


def build_qa_graph(llm: Optional[GeminiClient] = None, store: Optional[VectorStore] = None):
    llm = llm or GeminiClient()
    store = store or VectorStore()

    graph = StateGraph(AgentState)
    graph.add_node("qa_answer", lambda s: node_qa_answer(s, llm, store))
    graph.set_entry_point("qa_answer")
    graph.add_edge("qa_answer", END)
    return graph.compile()
