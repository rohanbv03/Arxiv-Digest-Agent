"""
Builds the structured executive briefing.

Context selection: each priority section gets its OWN fixed character
budget (config.SECTION_CHAR_BUDGET), rather than one global running budget
consumed in section order. The earlier version used a shared budget that
Introduction/Method could exhaust before Results/Limitations were even
considered -- exactly the failure mode where a long Method section could
silently starve the model of the paper's actual findings. A fixed
per-section cap guarantees every priority section contributes something.
Full multi-call map-reduce summarization (summarize each section
separately, then combine) would handle very long individual sections
better still; it's listed as a future upgrade in the README rather than
implemented here, since it multiplies LLM calls per paper and this
assessment targets a free tier.

Limitations vs. caveats: the model is explicitly asked to separate
limitations the *authors themselves* state from caveats *we* infer from
the method/results, rather than blending fabricated-sounding "(inferred)"
items into one list.
"""
from typing import Dict, List

from . import config
from .llm import GeminiClient
from .state import PaperMeta

PRIORITY_SECTION_KEYS = [
    "abstract", "introduction", "method", "methods", "methodology",
    "approach", "results", "experiments", "evaluation", "discussion",
    "conclusion", "conclusions", "limitations",
]


def _select_context(paper: PaperMeta, sections: Dict[str, str]) -> str:
    parts = [f"ABSTRACT:\n{paper.get('abstract', '')}"]
    seen = {"abstract"}
    for key in PRIORITY_SECTION_KEYS:
        if key in sections and key not in seen:
            seen.add(key)
            snippet = sections[key][: config.SECTION_CHAR_BUDGET]
            truncated_note = " [...truncated]" if len(sections[key]) > config.SECTION_CHAR_BUDGET else ""
            parts.append(f"{key.upper()}:\n{snippet}{truncated_note}")
    return "\n\n".join(parts)


BRIEFING_SCHEMA_HINT = """
{
  "title": string,
  "authors": [string],
  "arxiv_id": string,
  "published": string,
  "link": string,
  "summary": string,                // 1 paragraph, plain English, "why this paper matters"
  "problem_statement": string,
  "method": [string],               // bullet points
  "key_results": [string],
  "limitations": [string],          // ONLY limitations the paper's authors explicitly state. Empty list if they state none -- do NOT invent items here.
  "caveats": [string],              // OUR analysis: plausible weaknesses inferred from the method/results, kept separate from what the authors actually claimed
  "suggested_questions": [string]   // 3-5 follow-up questions a reader might ask
}
"""


def build_briefing(
    paper: PaperMeta,
    sections: Dict[str, str],
    parse_warning: str,
    llm: GeminiClient,
) -> Dict:
    context = _select_context(paper, sections)

    prompt = f"""
You are an expert research assistant producing an executive briefing for a
busy engineer deciding whether to read this paper in full.

PAPER METADATA:
- Title: {paper.get('title')}
- Authors: {', '.join(paper.get('authors', []))}
- arXiv ID: {paper.get('arxiv_id')}
- Published: {paper.get('published')}
- Link: {paper.get('abs_url')}

PAPER CONTENT (partial, extracted from PDF; each section capped independently):
{context}

{"NOTE: " + parse_warning if parse_warning else ""}

Produce a JSON object matching exactly this shape:
{BRIEFING_SCHEMA_HINT}

Rules:
- Base every claim only on the content above; do not invent results.
- "limitations" must contain ONLY things the paper itself states as a
  limitation. If the paper states none explicitly, return an empty list
  for "limitations" -- do not fill it with guesses.
- "caveats" is where YOUR inferred weaknesses go instead (e.g. "evaluated
  on a single dataset/domain"), clearly distinct from author claims.
- Keep "method" and "key_results" to concise bullet-style strings (not full paragraphs).
"""
    briefing = llm.generate_json(prompt)

    briefing.setdefault("title", paper.get("title", ""))
    briefing.setdefault("authors", paper.get("authors", []))
    briefing.setdefault("arxiv_id", paper.get("arxiv_id", ""))
    briefing.setdefault("published", paper.get("published", ""))
    briefing.setdefault("link", paper.get("abs_url", ""))
    briefing.setdefault("limitations", [])
    briefing.setdefault("caveats", [])
    if not briefing["limitations"] and not briefing["caveats"]:
        briefing["caveats"] = [
            "The authors do not explicitly discuss limitations in the "
            "sections available to the agent; no inferred caveats were "
            "generated to avoid overstating confidence in a guess."
        ]
    return briefing


def briefing_to_markdown(b: Dict) -> str:
    def bullets(items: List[str]) -> str:
        return "\n".join(f"- {i}" for i in items) if items else "- (none stated)"

    return f"""# {b.get('title', 'Untitled')}

**arXiv:** [{b.get('arxiv_id', '')}]({b.get('link', '')})  |  **Published:** {b.get('published', '')}
**Authors:** {', '.join(b.get('authors', []))}

## Summary
{b.get('summary', '')}

## Problem Statement
{b.get('problem_statement', '')}

## Method
{bullets(b.get('method', []))}

## Key Results
{bullets(b.get('key_results', []))}

## Limitations (stated by the authors)
{bullets(b.get('limitations', []))}

## Additional Caveats (inferred, not claimed by the authors)
{bullets(b.get('caveats', []))}

## Suggested Follow-up Questions
{bullets(b.get('suggested_questions', []))}
"""
