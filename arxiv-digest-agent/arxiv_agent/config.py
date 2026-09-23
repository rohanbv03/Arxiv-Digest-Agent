"""
Central place for configuration. Everything here can be overridden via
environment variables (see .env.example) so the agent can be re-tuned
without touching code.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys / models -------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# NOTE: Google retires Gemini model ids on a rolling basis (gemini-1.5-flash
# and text-embedding-004 are already gone). As of writing, gemini-2.5-flash
# is on the free tier but scheduled for shutdown 2026-10-16 -- check
# https://ai.google.dev/gemini-api/docs/models for the current lineup
# (e.g. gemini-3.1-flash-lite) and override via env vars if it's moved on.
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

# --- storage -------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(os.getcwd(), ".chroma_store"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "outputs"))
PDF_CACHE_DIR = os.getenv("PDF_CACHE_DIR", os.path.join(os.getcwd(), ".pdf_cache"))

# --- chunking --------------------------------------------------------------
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", 350))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", 60))

# --- retrieval ---------------------------------------------------------
MAX_ARXIV_CANDIDATES = int(os.getenv("MAX_ARXIV_CANDIDATES", 8))
TOP_K_QA_CHUNKS = int(os.getenv("TOP_K_QA_CHUNKS", 5))

# Chroma returns cosine *distance* (0 = identical, 2 = opposite). Above this
# distance for the single best chunk, we treat the question as unanswerable
# from the paper rather than risk a hallucinated answer.
GROUNDING_DISTANCE_CEILING = float(os.getenv("GROUNDING_DISTANCE_CEILING", 0.9))

# --- pdf parsing safety valve -------------------------------------------
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 60))  # cap for very long papers

# --- summarization ---------------------------------------------------------
# Fixed per-section character budget for the briefing prompt. Each priority
# section (Method, Results, Limitations, ...) gets this many characters
# independently, rather than all sections sharing one global budget that a
# long early section (e.g. Method) could exhaust before Results is reached.
SECTION_CHAR_BUDGET = int(os.getenv("SECTION_CHAR_BUDGET", 3000))
