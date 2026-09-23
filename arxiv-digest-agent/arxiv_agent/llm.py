"""
Thin wrapper around the `google-genai` SDK (the current package; the older
`google-generativeai` is deprecated) so the rest of the codebase never
imports the SDK directly -- makes it a one-file swap to Groq/Ollama later.
"""
import json
import re
import time
from typing import List, Optional

from google import genai
from google.genai import types

from . import config


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


class GeminiClient:
    def __init__(self, model_name: Optional[str] = None, embed_model: Optional[str] = None):
        if not config.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add "
                "a free Gemini API key from https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.model_name = model_name or config.GEN_MODEL
        self.embed_model = embed_model or config.EMBED_MODEL

    def generate_text(self, prompt: str, temperature: float = 0.3) -> str:
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return (resp.text or "").strip()

    def generate_json(self, prompt: str, temperature: float = 0.2) -> dict:
        """
        Asks for JSON-only output and parses it. Retries once with a
        stricter reminder if the first parse fails -- the model occasionally
        wraps JSON in prose or a code fence despite instructions.
        """
        full_prompt = (
            prompt
            + "\n\nRespond with ONLY valid JSON. No markdown fences, no "
            "commentary before or after the JSON object."
        )
        last_err = None
        for attempt in range(2):
            raw = self.generate_text(full_prompt, temperature=temperature)
            try:
                return json.loads(_strip_code_fence(raw))
            except json.JSONDecodeError as e:
                last_err = e
                full_prompt += (
                    "\n\nYour previous response was not valid JSON. Return "
                    "ONLY the JSON object, nothing else."
                )
                time.sleep(0.5)
        raise ValueError(f"Model did not return valid JSON after retries: {last_err}")

    def embed_texts(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """
        Batches embedding calls (the SDK accepts a list for `contents`)
        instead of one request per chunk -- a 40-page paper can produce
        100+ chunks, and one-request-per-chunk burns free-tier RPM/RPD
        quota fast. `batch_size` is conservative; raise it if your
        account's per-request payload limit allows more.
        """
        if not texts:
            return []
        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.client.models.embed_content(
                model=self.embed_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            vectors.extend(e.values for e in result.embeddings)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.embed_model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values
