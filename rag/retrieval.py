"""Simple retrieval pipeline for the RAG lab using Chroma.

We assume that `ingestion.ingest_chunks` has already populated the
collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import chromadb
from chromadb.config import Settings

from .config import CHROMA_DIR, DEFAULT_TOP_K


@dataclass
class RetrievedChunk:
  id: str
  text: str
  document_path: str
  source: str
  score: float


def get_collection(collection_name: str = "bsk_rag"):
  """Return a Chroma collection (creating client from CHROMA_DIR)."""

  client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
  return client.get_or_create_collection(name=collection_name)


def query_chunks(query: str, top_k: int = DEFAULT_TOP_K, collection_name: str = "bsk_rag") -> List[RetrievedChunk]:
  """Retrieve top_k most relevant chunks for a query.

  Returns a list of RetrievedChunk objects suitable for building RAG
  prompts and UI source citations.
  """

  collection = get_collection(collection_name)

  if not query.strip():
    return []

  result = collection.query(query_texts=[query], n_results=top_k)

  # Chroma returns dict with keys: ids, documents, metadatas, distances/similarities
  ids = result.get("ids", [[]])[0]
  docs = result.get("documents", [[]])[0]
  metas = result.get("metadatas", [[]])[0]
  distances = result.get("distances", [[]]) or result.get("embeddings", [[]])
  scores = distances[0] if distances else [0.0] * len(ids)

  retrieved: List[RetrievedChunk] = []
  for idx, chunk_id in enumerate(ids):
    meta = metas[idx] or {}
    text = docs[idx] or ""
    score = float(scores[idx]) if idx < len(scores) else 0.0
    retrieved.append(
      RetrievedChunk(
        id=str(chunk_id),
        text=text,
        document_path=str(meta.get("document_path", "")),
        source=str(meta.get("source", "")),
        score=score,
      )
    )

  return retrieved
