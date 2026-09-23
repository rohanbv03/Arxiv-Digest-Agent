"""
The single shared state object that flows through the graph(s).

Two separate compiled graphs share this schema:
  - the *pipeline* graph (query understanding -> ... -> summarize)
  - the *qa* graph (one turn: question in, grounded answer out)

Splitting them (rather than one graph with an input()-calling QA node)
keeps orchestration logic out of the CLI's job (reading input) and the
graph's job (routing state) -- see graph.py for the split and cli.py for
how the two are driven from a normal interactive loop.

Persistence note (this used to overclaim, see README "Persistence" for the
corrected version): AgentState itself, and the LangGraph MemorySaver
checkpointer, are both in-memory and gone when the process exits. What
actually survives a restart is the Chroma vector store and the cached PDF
on disk -- which is exactly what node_chunk_and_embed's index-reuse check
relies on.
"""
from typing import TypedDict, List, Dict, Optional, Any


class PaperMeta(TypedDict, total=False):
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    pdf_url: str
    abs_url: str
    categories: List[str]
    published: str


class Chunk(TypedDict, total=False):
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    section: str


class RetrievedChunk(TypedDict, total=False):
    chunk_index: int
    distance: float
    page_start: int
    page_end: int
    section: str


class QATurn(TypedDict, total=False):
    question: str
    answer: str
    grounded: bool
    sources: List[RetrievedChunk]


class AgentState(TypedDict, total=False):
    # --- input / intent -------------------------------------------------
    user_input: str
    intent: str  # "id" | "topic"
    normalized_id: Optional[str]
    search_query: Optional[str]  # cleaned-up query actually sent to arXiv

    # --- retrieval --------------------------------------------------------
    candidates: List[PaperMeta]
    selected_paper: Optional[PaperMeta]

    # --- parsing (page-aware) ----------------------------------------
    pages: List[str]              # raw text per page (index 0 = page 1)
    page_sections: List[str]      # dominant section label per page
    sections: Dict[str, str]      # concatenated text per section (for briefing)
    references: List[str]
    parse_warning: Optional[str]
    used_abstract_fallback: bool

    # --- chunk / embed ------------------------------------------------
    collection_name: str
    num_chunks: int
    index_reused: bool

    # --- summarize --------------------------------------------------------
    briefing: Optional[Dict[str, Any]]

    # --- qa (single-turn graph state) --------------------------------
    pending_question: Optional[str]
    qa_answer: Optional[str]
    qa_grounded: Optional[bool]
    qa_sources: List[RetrievedChunk]
    qa_history: List[QATurn]

    # --- control / errors ---------------------------------------------
    error: Optional[str]
    error_kind: Optional[str]  # "no_results" | "api_failure" | "parse_failure"
