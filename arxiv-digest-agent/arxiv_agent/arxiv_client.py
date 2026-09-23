"""
Thin wrapper around the official arXiv Atom API (export.arxiv.org/api/query).
No scraping, no third-party arxiv package required -- just requests + stdlib
XML parsing, so the surface area we depend on is small and easy to reason
about when something breaks.
"""
import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests

from .state import PaperMeta

ATOM_NS = "{http://www.w3.org/2005/Atom}"
API_URL = "http://export.arxiv.org/api/query"

# Matches things like 2401.12345, 2401.12345v2, or a full arxiv.org URL.
ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE
)

# Filler phrases stripped from natural-language topic queries before they're
# sent to arXiv's search endpoint -- "recent work on X" and "X" retrieve
# very differently once you're matching against titles/abstracts rather
# than a human reading the sentence.
_FILLER_PATTERNS = [
    r"\brecent (work|papers|research|advances?)\s+(on|in|about)\b",
    r"\b(papers?|research|work|studies)\s+(on|about|regarding)\b",
    r"\b(what is|what are)\b",
    r"\b(the )?latest\b",
    r"\bfor (llms?|large language models?)\b",
]


class ArxivRetrievalError(Exception):
    """Raised on network/API failure, as distinct from a clean zero-results search."""


def extract_arxiv_id(text: str) -> Optional[str]:
    """Return a bare arXiv id (no version suffix) if `text` looks like an id/URL."""
    match = ARXIV_ID_RE.search(text.strip())
    if not match:
        return None
    raw_id = match.group(1)
    return re.sub(r"v\d+$", "", raw_id)


def clean_topic_query(text: str) -> str:
    """
    Lightweight normalization of a natural-language topic into something
    closer to a keyword query, without a full LLM call. Strips common
    framing phrases and trims to the substantive terms. This is a
    heuristic, not NLP -- see README for why a full query-rewriting LLM
    call was judged not worth the extra latency/cost for this use case.
    """
    cleaned = text.strip()
    for pattern in _FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,-")
    return cleaned or text.strip()


def _parse_entry(entry: ET.Element) -> PaperMeta:
    def txt(tag: str, ns: str = ATOM_NS) -> str:
        el = entry.find(f"{ns}{tag}")
        return (el.text or "").strip() if el is not None else ""

    full_id_url = txt("id")
    arxiv_id = extract_arxiv_id(full_id_url) or full_id_url

    authors = [
        (a.find(f"{ATOM_NS}name").text or "").strip()
        for a in entry.findall(f"{ATOM_NS}author")
        if a.find(f"{ATOM_NS}name") is not None
    ]

    pdf_url = ""
    for link in entry.findall(f"{ATOM_NS}link"):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href", "")
    if not pdf_url and full_id_url:
        pdf_url = full_id_url.replace("/abs/", "/pdf/") + ".pdf"

    categories = [
        c.get("term", "") for c in entry.findall(f"{ATOM_NS}category") if c.get("term")
    ]

    return PaperMeta(
        arxiv_id=arxiv_id,
        title=" ".join(txt("title").split()),
        authors=authors,
        abstract=" ".join(txt("summary").split()),
        pdf_url=pdf_url,
        abs_url=full_id_url,
        categories=categories,
        published=txt("published")[:10],
    )


def _get(params: dict) -> ET.Element:
    try:
        resp = requests.get(API_URL, params=params, timeout=20)
        resp.raise_for_status()
    except requests.Timeout as e:
        raise ArxivRetrievalError("arXiv API timed out -- try again in a moment.") from e
    except requests.RequestException as e:
        raise ArxivRetrievalError(f"Could not reach the arXiv API: {e}") from e

    try:
        return ET.fromstring(resp.text)
    except ET.ParseError as e:
        raise ArxivRetrievalError(f"arXiv returned an unparseable response: {e}") from e


def fetch_by_id(arxiv_id: str) -> List[PaperMeta]:
    root = _get({"id_list": arxiv_id})
    entries = root.findall(f"{ATOM_NS}entry")
    return [_parse_entry(e) for e in entries]


def search_arxiv(query: str, max_results: int = 8) -> List[PaperMeta]:
    root = _get(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    entries = root.findall(f"{ATOM_NS}entry")
    return [_parse_entry(e) for e in entries]
