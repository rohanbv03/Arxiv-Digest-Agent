# Autonomous arXiv Paper Digest & Grounded QA Agent

A LangGraph agent that takes a research topic or an arXiv ID/URL, retrieves
and parses the paper, produces a structured executive briefing, and answers
follow-up questions grounded in the paper's own text via RAG, with
page/section citations.

Topic search is intentionally **not** fully autonomous: results are ranked
automatically, but a human confirms which candidate paper to process. A
direct arXiv ID/URL is fully automatic end to end.

## Architecture

Two separate compiled graphs, not one graph with UI logic mixed into a node:

```
PIPELINE GRAPH (runs once per paper)

query_understanding
        |
arxiv_retrieval --(zero candidates)-------> END (error_kind="no_results")
        |        --(network/API failure)--> END (error_kind="api_failure")
        |
selection_ranking   (ranks candidates by embedding similarity; the actual
        |             pick is made by a callback the caller supplies --
        |             interactive input() in the CLI, "take top match" in
        |             tests/non-interactive mode -- not hardcoded into the node)
        |
fetch_and_parse --(PDF unrecoverable: corrupt/scanned)--> falls back to
        |          abstract-only mode with a warning recorded in state
        |
chunk_and_embed  --> page+section-aware chunks, embedded (batched) into a
        |            local on-disk Chroma collection. If a collection for
        |            this arXiv ID already has vectors, embedding is
        |            skipped entirely (index_reused=True).
        |
summarize --> structured JSON + Markdown briefing, saved to ./outputs/
        |
       END


QA GRAPH (one invocation per question, driven by the CLI's own loop)

qa_answer --> retrieve top-k chunks --> distance gate --> LLM verifies
              support + drafts answer in one JSON call --> END
```

**Why two graphs instead of one with a looping QA node containing
`input()`:** a graph node calling Python's `input()` mixes orchestration
(routing state) with a CLI-specific concern (reading a line from stdin),
and makes the graph unusable non-interactively (tests, a future API
server, `--questions` batch mode). The QA graph takes `pending_question`
and `collection_name` and returns an answer; the CLI's own `while True`
loop calls `qa_graph.invoke(...)` once per question.

**State shape** (`AgentState`, TypedDict, `arxiv_agent/state.py`):
`user_input`, `intent`, `search_query` (cleaned topic query), `candidates`,
`selected_paper`, `pages`/`page_sections` (per-page text + section label,
enabling citations), `sections` (concatenated per-section text, used for
the briefing prompt), `references`, `parse_warning`, `collection_name`,
`num_chunks`, `index_reused`, `briefing`, `pending_question`/`qa_answer`/
`qa_grounded`/`qa_sources`, `qa_history`, `error`/`error_kind`.

### Persistence, precisely

- **`AgentState` itself and the LangGraph `MemorySaver` checkpointer are
  in-memory only** and are gone when the process exits. They exist so a
  single run's state flows cleanly node-to-node (and so a long-lived host
  process could keep multiple concurrent sessions apart by `thread_id`),
  not to survive a restart.
- **What actually survives a restart** is the downloaded PDF
  (`.pdf_cache/`) and the embedded chunks (`.chroma_store/`, one Chroma
  collection per arXiv ID). `node_chunk_and_embed` checks
  `store.exists(collection_name)` before embedding anything; if a prior
  run already indexed this exact paper, it reuses that collection and
  skips re-embedding rather than deleting and re-adding.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed to run the tests
cp .env.example .env   # then edit .env and add a free Gemini API key
```

Get a free key at https://aistudio.google.com/apikey (no card required).

## Run

```bash
python cli.py "recent work on KV-cache compression for LLMs"
python cli.py 2401.12345
python cli.py https://arxiv.org/abs/2401.12345

# non-interactive (no input() prompts -- good for a demo recording):
python cli.py 2401.12345 --questions "What is the main contribution?" "What dataset do they evaluate on?"
```

The briefing is printed to the console and saved to
`outputs/<arxiv_id>_briefing.{json,md}`. Without `--questions`, the CLI
then drops into an interactive QA loop (`exit` to quit); each answer is
followed by a `Sources:` block listing the page(s)/section it drew from,
not just an opaque chunk number.

### Tests

```bash
pytest
```

32 tests, all offline (no arXiv/Gemini calls) via fakes/monkeypatching:
arXiv ID/URL extraction and topic-query cleanup, chunking (including
page/section tracking), the PDF section/reference-splitting heuristics,
the vector store's exists/count/add/query behavior (including the
empty-collection case), QA grounding (far-distance short-circuit, empty
retrieval, LLM-verified supported/unsupported verdicts), and the graph's
conditional routing for zero-results / API-failure / success (via
monkeypatched arXiv calls) plus both graphs compiling.

### Example run (illustrative)

```
$ python cli.py "KV-cache compression for LLMs"

Found 6 candidate papers:
  [0] (0.81) <Paper A title>  (24xx.xxxxx, 2024-xx-xx)
  [1] (0.74) <Paper B title>  (24xx.xxxxx, 2024-xx-xx)
  ...
Pick a paper to summarize [0-5, default 0]: 0

======================================================================
# <Paper A title>
...
## Limitations (stated by the authors)
- ...
## Additional Caveats (inferred, not claimed by the authors)
- ...
======================================================================
(saved to outputs/24xx.xxxxx_briefing.json and outputs/24xx.xxxxx_briefing.md)

Ask a question about this paper (or type 'exit' to quit): What eviction
policy do they use for the KV cache?

<grounded answer>
Sources:
  - p.6 -- method

Ask a question about this paper (or type 'exit' to quit): Does it run on TPUs?

That doesn't appear to be covered in this paper based on what I could
retrieve, so I won't guess.

Ask a question about this paper (or type 'exit' to quit): exit
```

*(Titles/IDs are placeholders -- capturing a real transcript requires
network access to arxiv.org and the Gemini API, which this build
environment doesn't have. Run it locally and replace this block with your
actual output/recording before submitting -- see "Known limitations".)*

## Design Decisions & Tradeoffs

**Orchestration -- two LangGraph graphs, not one.** See "Architecture"
above. Splitting the once-per-paper pipeline from the once-per-question QA
turn keeps `input()` (a CLI concern) out of graph nodes entirely, and
makes `pending_question` load-bearing instead of a declared-but-unused
field.

**Selection -- embedding similarity + human-in-the-loop, via a callback.**
For a topic search, ranking candidates by cosine similarity between the
query embedding and each abstract embedding is cheap (no extra generation
call) and transparent (the score is shown). The actual pick is made by a
`SelectionCallback` the caller injects -- `input()`-based in the CLI,
"take the top match" in tests/`--questions` mode -- so the node itself has
no I/O and is directly unit-testable.

**Chunking -- page/section-aware sliding window, not naive fixed windows.**
Chunks are built by flattening (word, page, section) triples across the
whole paper and windowing over that, so every chunk carries a
`page_start`/`page_end`/`section` -- which is what lets a QA answer say
"p.7, Results" instead of "chunk 14". The window itself is still a fixed
word count with overlap (not fully section-bounded), which stays robust to
messy PDF extraction (misdetected headers, multi-column bleed); the
overlap (60/350 words) mitigates the main failure mode of a plain window,
an answer sentence straddling a boundary. Fully section-bounded chunking
(never crossing a section boundary) is the natural next step and is listed
below.

**Grounding -- a distance gate AND an LLM evidence-verification step, not
either alone.** A raw cosine-distance threshold on the best retrieved
chunk is a cheap pre-filter but not a reliable "does the paper actually
answer this" detector -- a chunk can be topically close without containing
the specific fact asked for. So retrieval is followed by a single
JSON-generation call that asks the model for a `supported: bool` verdict
plus which chunk indices it actually used, and the CLI/graph trust that
verdict over just hoping the model followed a "don't guess" instruction.
This is one JSON call per question (same cost as a plain text-generation
call would have been), not two.

**Summarization -- per-section fixed budget, not one global running
budget.** The first draft of this agent walked priority sections in order
and stopped once a shared character budget ran out, which meant a long
Method section could silently exclude Results/Limitations from ever
reaching the model -- a real correctness risk for a rubric that explicitly
grades "key results" and "limitations". Giving each priority section its
own fixed budget (`SECTION_CHAR_BUDGET`, default 3000 chars) guarantees
every section present in the paper contributes something. True multi-call
map-reduce summarization (summarize each section separately with its own
full text, then combine) would handle individual very-long sections even
better, at the cost of several extra LLM calls per paper; not implemented
here to stay call-frugal on a free tier, but noted as the next upgrade.

**Limitations vs. caveats, kept separate.** Early on, the model was asked
to "infer plausible limitations" when the paper stated none, and prefix
them `(inferred)`. In practice that's dangerous: a paper simply describing
its evaluation setup (e.g. "we test on dataset X") could get turned into
an invented-sounding "may not generalize to other datasets" bullet
presented as if it were the same kind of claim as an author-stated
limitation. The briefing now has two separate fields -- `limitations`
(only what the authors themselves state; empty list if they state none)
and `caveats` (our own inferred weaknesses, clearly labeled as analysis,
not attributed to the authors).

**arXiv error handling -- distinguish "no results" from "API failure".**
Network/timeout/malformed-XML errors raise `ArxivRetrievalError` and route
to `END` with `error_kind="api_failure"`; a clean zero-hit search routes
to `END` with `error_kind="no_results"`. Same routing target, different
message -- a user retrying a bad connection needs a different signal than
a user whose topic just doesn't exist on arXiv.

**Topic query cleanup -- regex normalization, not an LLM call.**
`arxiv_client.clean_topic_query` strips common framing phrases ("recent
work on", "papers about", "the latest", ...) before the query hits
arXiv's `all:` search, since arXiv is matching keywords against
titles/abstracts, not parsing a sentence. A full LLM-based query rewrite
would likely do better on unusual phrasings, but costs a generation call
and adds latency for a search step that runs before the user has even
picked a paper; the regex approach was judged good-enough for the common
"recent work on X" / "papers about X" framings this assignment's own
example uses.

**Embedding calls are batched.** The original per-chunk embedding loop
issues one API request per chunk -- for a 100-chunk paper, 100 requests,
which burns free-tier RPM/RPD fast. `GeminiClient.embed_texts` now batches
into groups of `batch_size` (default 20) using the SDK's list-`contents`
support.

**Index reuse is real, not cosmetic.** `VectorStore.exists()` checks
whether a Chroma collection for this arXiv ID already has vectors before
`node_chunk_and_embed` does any parsing-adjacent work; if so, it skips
straight to using the existing collection (`index_reused=True` in state)
instead of unconditionally deleting-then-re-adding.

**What I'd do differently with more time:**
- Fully section-bounded chunks (never straddling a section boundary), not just section-*labeled* fixed windows.
- True map-reduce summarization for individual very-long sections.
- A proper re-ranker (even a cheap cross-encoder) instead of raw abstract-embedding cosine similarity for candidate selection.
- A local embedding backend option (e.g. `sentence-transformers`) so RAG indexing doesn't consume Gemini quota at all, keeping the free-tier budget entirely for generation.
- Persisting `AgentState` itself (not just the vector store) via LangGraph's `SqliteSaver`, so a QA session's history survives a process restart, not just an in-process loop.
- Basic OCR fallback (e.g. `pytesseract`) for scanned PDFs instead of refusing them.
- A `--demo` mode with scripted output timing, for a cleaner recording (`--questions` covers the non-interactive part already).

## Rate limits (free tier, Gemini API, as of the time this was built)

Google retires and reprices Gemini model ids frequently -- `gemini-1.5-flash`
and `text-embedding-004` (older stable defaults) have already been shut
down. This code defaults to `gemini-2.5-flash` and `gemini-embedding-001`,
which were free-tier-available at time of writing, but `gemini-2.5-flash`
is itself scheduled for shutdown in mid-October 2026. **If you hit a
"model not found" error, check https://ai.google.dev/gemini-api/docs/models
for the current lineup and set `GEN_MODEL`/`EMBED_MODEL` in `.env`** -- no
code changes needed. Free tier is also request-per-minute and
request-per-day limited (historically on the order of 10-15 RPM / a few
hundred-to-1500 RPD for Flash-tier models); if you see `429
RESOURCE_EXHAUSTED`, wait a minute and retry, or lower
`TOP_K_QA_CHUNKS`/`MAX_PDF_PAGES` to send fewer/smaller requests. Embedding
calls are batched (see above) specifically to reduce how often you'll hit
this.

## Known limitations

- Not tested against the live arXiv/Gemini APIs in the environment this
  was built in (no outbound network access to arxiv.org or
  generativelanguage.googleapis.com there). Everything network-independent
  -- arXiv ID/URL regex extraction, topic-query cleanup, the section/page
  parsing heuristics, chunking, vector-store exists/add/query behavior,
  QA grounding logic (via fakes), and both graphs compiling/routing
  correctly -- is covered by the 32 tests in `tests/` (`pytest`, all
  passing). **Please run it end-to-end against the real APIs and fix
  anything that surfaces** -- a live PDF's exact formatting or a real
  Gemini JSON response occasionally deviating from the schema are the most
  likely sources of surprises.
- Section detection is regex/heading-based and will miss papers that don't
  use conventional section names, or that put headings in a font/format
  PyMuPDF's plain-text extraction doesn't preserve as a standalone line --
  chunks from such a paper will carry `section: "unknown"` rather than a
  wrong label.
- No OCR -- scanned/image-only PDFs fall back to abstract-only mode rather
  than being read.
- Reference extraction is a best-effort regex split, not a real citation
  parser (won't handle every bibliography style).
- Topic-query cleanup is regex-based and won't handle every phrasing as
  well as an LLM rewrite would (see Design Decisions).
- Single-user, single-process, no auth/deployment -- intentionally, per the
  assessment's "out of scope" list.

## Deliverables checklist (from the assessment)

- [x] Working code, this repo
- [x] README with architecture, setup, design tradeoffs (this file)
- [x] Tests (`tests/`, `pytest`)
- [ ] Example run screen recording/gif -- record locally after you confirm it runs
- [ ] 4-minute video reflection -- record separately; not something I can produce for you
