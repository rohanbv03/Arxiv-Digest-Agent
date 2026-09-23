"""
Local, on-disk vector store (Chroma's PersistentClient -- no server, no
cloud account). We pass embeddings in explicitly (computed via
llm.GeminiClient) rather than using Chroma's built-in embedding function,
so the embedding call sites stay visible and swappable in one place.
"""
import re
from typing import Dict, List

import chromadb

from . import config
from .state import Chunk, RetrievedChunk


def sanitize_collection_name(arxiv_id: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", arxiv_id)
    return f"paper_{name}"[:63]


class VectorStore:
    def __init__(self, persist_dir: str = config.CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)

    def get_or_create(self, collection_name: str):
        return self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def exists(self, collection_name: str) -> bool:
        """True if this paper already has a non-empty index on disk."""
        try:
            collection = self.client.get_collection(collection_name)
        except Exception:
            return False
        return collection.count() > 0

    def count(self, collection_name: str) -> int:
        try:
            return self.client.get_collection(collection_name).count()
        except Exception:
            return 0

    def add_chunks(
        self,
        collection_name: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        arxiv_id: str = "",
    ):
        collection = self.get_or_create(collection_name)
        ids = [f"{collection_name}_{c['chunk_index']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "chunk_index": c["chunk_index"],
                "page_start": c.get("page_start", 1),
                "page_end": c.get("page_end", 1),
                "section": c.get("section", "unknown"),
                "arxiv_id": arxiv_id,
            }
            for c in chunks
        ]
        collection.add(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )
        return collection

    def query(self, collection_name: str, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        collection = self.get_or_create(collection_name)
        if collection.count() == 0:
            return []
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        out: List[RetrievedChunk] = []
        for doc, dist, meta in zip(docs, distances, metadatas):
            out.append(
                {
                    "text": doc,
                    "distance": dist,
                    "chunk_index": meta.get("chunk_index"),
                    "page_start": meta.get("page_start"),
                    "page_end": meta.get("page_end"),
                    "section": meta.get("section"),
                }
            )
        return out
