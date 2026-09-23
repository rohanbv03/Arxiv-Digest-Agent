"""
Download + parse a paper PDF. Kept deliberately defensive: arXiv PDFs vary a
lot (scanned older papers, huge appendices, unusual layouts), and the graph
node calling this module is expected to fall back gracefully rather than
crash the whole run (see graph.py: node_fetch_and_parse).

Page-aware: text is kept per-page (not flattened immediately) so that
downstream chunking can attach a page range to every chunk, which is what
lets QA answers cite "page 7" instead of an opaque chunk index.
"""
import re
from typing import Dict, List, Tuple

import pymupdf as fitz  # PyMuPDF (new import name; `fitz` alias is deprecated)
import requests

from . import config

COMMON_HEADERS = [
    "abstract", "introduction", "related work", "background",
    "method", "methods", "methodology", "approach",
    "experiments", "experimental setup", "results", "evaluation",
    "discussion", "conclusion", "conclusions", "limitations",
    "acknowledgments", "acknowledgements", "references",
]

HEADER_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(" + "|".join(COMMON_HEADERS) + r")\s*$",
    re.IGNORECASE,
)


class PdfParseError(Exception):
    pass


def download_pdf(url: str, arxiv_id: str) -> str:
    import os
    os.makedirs(config.PDF_CACHE_DIR, exist_ok=True)
    dest = os.path.join(config.PDF_CACHE_DIR, f"{arxiv_id}.pdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise PdfParseError(f"Failed to download PDF: {e}") from e

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def _detect_header(line: str) -> str:
    m = HEADER_RE.match(line.strip())
    return m.group(1).lower() if m else ""


def _split_sections_with_pages(pages: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Walk pages line-by-line. Returns:
      - sections: header-name -> concatenated text (for briefing context)
      - page_sections: one label per page = the section active by the
        *end* of that page (a page can start mid-section and end in a
        new one; we tag it with whichever section owns most of its
        content by using the last header seen on/before that page).
    """
    sections: Dict[str, List[str]] = {}
    page_sections: List[str] = []
    current = "preamble"
    sections.setdefault(current, [])

    for page_text in pages:
        for line in page_text.split("\n"):
            header = _detect_header(line)
            if header:
                current = header
                sections.setdefault(current, [])
            else:
                sections[current].append(line)
        page_sections.append(current)

    joined = {k: "\n".join(v).strip() for k, v in sections.items()}
    return {k: v for k, v in joined.items() if v}, page_sections


def _extract_references(sections: Dict[str, str]) -> List[str]:
    ref_blob = sections.get("references", "")
    if not ref_blob:
        return []
    parts = re.split(r"\n(?=\[?\d{1,3}\]?[\.\)]?\s)", ref_blob)
    return [p.strip().replace("\n", " ") for p in parts if p.strip()][:200]


# Kept for direct unit-testing / backward compatibility: same heuristic as
# _split_sections_with_pages but operating on one flat string.
def _split_sections(full_text: str) -> Dict[str, str]:
    sections, _ = _split_sections_with_pages([full_text])
    return sections


def parse_pdf(path: str) -> Tuple[List[str], List[str], Dict[str, str], List[str], str]:
    """
    Returns (pages, page_sections, sections, references, warning).
      pages         -- raw text of each parsed page, in order
      page_sections -- section label per page (same length as `pages`)
      sections      -- concatenated text per section, across all pages
      references    -- best-effort parsed reference list
      warning       -- "" on a clean parse, else a note (e.g. truncation)
                        that gets folded into the briefing's Limitations.
    """
    warning = ""
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise PdfParseError(f"PyMuPDF could not open the file (possibly corrupt): {e}") from e

    n_pages = doc.page_count
    if n_pages == 0:
        raise PdfParseError("PDF has zero pages.")

    pages_to_read = min(n_pages, config.MAX_PDF_PAGES)
    if pages_to_read < n_pages:
        warning = (
            f"Paper has {n_pages} pages; only the first {pages_to_read} were "
            f"parsed to keep processing bounded. Content beyond that point "
            f"(often appendices) was not indexed."
        )

    pages: List[str] = []
    total_chars = 0
    for i in range(pages_to_read):
        page_text = doc[i].get_text("text")
        pages.append(page_text)
        total_chars += len(page_text)
    doc.close()

    if total_chars < 200:
        raise PdfParseError(
            "Extracted text is suspiciously short -- this PDF is likely "
            "scanned/image-based with no embedded text layer (OCR not "
            "implemented in this assessment build)."
        )

    sections, page_sections = _split_sections_with_pages(pages)
    references = _extract_references(sections)
    return pages, page_sections, sections, references, warning
