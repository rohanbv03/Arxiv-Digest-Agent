"""
RAG answer generation with two independent grounding gates:

  1. A hard cosine-distance threshold on the *best* retrieved chunk. If
     nothing retrieved is close enough, the LLM is never called at all --
     we return a fixed "not covered" response. Cheap, but a similarity
     score alone is not a reliable "does the paper actually answer this"
     detector (a chunk can be topically close without containing the
     answer), so it's a pre-filter, not the only check.
  2. An LLM evidence-verification step: rather than just instructing the
     model to "only use the context" and trusting it, we ask it to return
     a structured verdict -- `supported: bool` plus which chunks it
     actually used -- in the same JSON call that produces the answer. If
     it says the context doesn't support an answer, we surface that
     instead of whatever text it drafted.

This costs one JSON-generation call per question (same as a plain
generate_text call would have), not two, by asking for the verdict and
the answer together.
"""
from typing import List, Tuple

from . import config
from .llm import GeminiClient
from .state import RetrievedChunk
from .vector_store import VectorStore

NOT_COVERED_MESSAGE = (
    "That doesn't appear to be covered in this paper based on what I could "
    "retrieve, so I won't guess. Try rephrasing, or ask about the "
    "abstract/method/results directly."
)


def _format_chunk(rc: RetrievedChunk) -> str:
    loc = f"page {rc['page_start']}" if rc["page_start"] == rc["page_end"] else f"pages {rc['page_start']}-{rc['page_end']}"
    return f"[chunk {rc['chunk_index']} | {loc} | section: {rc.get('section', 'unknown')}]\n{rc['text']}"


def answer_question(
    question: str,
    collection_name: str,
    llm: GeminiClient,
    store: VectorStore,
    top_k: int = config.TOP_K_QA_CHUNKS,
) -> Tuple[str, bool, List[RetrievedChunk]]:
    """Returns (answer_text, grounded, sources_used)."""
    query_emb = llm.embed_query(question)
    retrieved = store.query(collection_name, query_emb, top_k)

    if not retrieved:
        return NOT_COVERED_MESSAGE, False, []

    best_distance = retrieved[0]["distance"]
    if best_distance is not None and best_distance > config.GROUNDING_DISTANCE_CEILING:
        # Gate 1: don't even spend a generation call if nothing retrieved
        # is plausibly relevant.
        return NOT_COVERED_MESSAGE, False, []

    context = "\n\n".join(_format_chunk(rc) for rc in retrieved)

    prompt = f"""
You are answering a question using ONLY the context chunks below, which
are excerpts from a single research paper.

CONTEXT:
{context}

QUESTION: {question}

Decide whether the context actually supports an answer (topical closeness
is not enough -- the specific fact asked for must be present). Respond
with ONLY this JSON shape:
{{
  "supported": boolean,
  "answer": string,          // if supported: a direct answer grounded in the context. if not supported: a one-sentence note that the paper doesn't cover this.
  "used_chunk_indices": [int]  // which chunk numbers you actually drew on; empty if not supported
}}
"""
    try:
        result = llm.generate_json(prompt, temperature=0.1)
    except ValueError:
        # Model didn't return parseable JSON even after a retry -- fail
        # closed rather than risk an ungrounded free-text answer.
        return NOT_COVERED_MESSAGE, False, []

    supported = bool(result.get("supported", False))
    answer = str(result.get("answer", "")).strip() or NOT_COVERED_MESSAGE

    if not supported:
        return answer, False, []

    used_indices = set(result.get("used_chunk_indices", []) or [])
    sources = [rc for rc in retrieved if rc["chunk_index"] in used_indices] or retrieved[:1]
    return answer, True, sources


def format_sources(sources: List[RetrievedChunk]) -> str:
    if not sources:
        return ""
    lines = []
    for rc in sources:
        loc = f"p.{rc['page_start']}" if rc["page_start"] == rc["page_end"] else f"pp.{rc['page_start']}-{rc['page_end']}"
        lines.append(f"  - {loc} -- {rc.get('section', 'unknown')}")
    return "Sources:\n" + "\n".join(lines)
