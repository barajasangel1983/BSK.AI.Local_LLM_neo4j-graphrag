"""Simple ingestion pipeline for the RAG lab using Chroma.

Responsibility:
- Read text files from DATA_DIR.
- Chunk them into overlapping segments.
- Create/update a Chroma collection with embeddings.

This is intentionally minimal so you can understand each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import chromadb
from chromadb.config import Settings

from .config import DATA_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
  """A single text chunk ready for embedding and storage."""

  id: str
  document_path: Path
  text: str


def _iter_text_files(root: Path) -> Iterable[Path]:
  """Yield all .txt files under the given root directory."""

  if not root.exists():
    return []

  return root.rglob("*.txt")


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
  """Naive character-based chunking with fixed overlap.

  This is simple but good enough for the first RAG lab. Later we can
  switch to token-based chunking.
  """

  if chunk_size <= 0:
    return [text]

  chunks: List[str] = []
  start = 0
  length = len(text)

  while start < length:
    end = min(start + chunk_size, length)
    chunks.append(text[start:end])
    if end == length:
      break
    start = max(0, end - overlap)

  return chunks


def build_chunks() -> List[Chunk]:
  """Load all text files under DATA_DIR and return their chunks.

  Each chunk gets an ID that includes the file path and index.
  """

  chunks: List[Chunk] = []

  for path in _iter_text_files(DATA_DIR):
    text = path.read_text(encoding="utf-8", errors="ignore")
    raw_chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    for idx, chunk_text in enumerate(raw_chunks):
      chunk_id = f"{path.relative_to(DATA_DIR)}::{idx}"
      chunks.append(Chunk(id=chunk_id, document_path=path, text=chunk_text))

  return chunks


def ingest_chunks(
  collection_name: str = "bsk_rag",
  owner_user_id: str = "admin",
  visibility: str = "private",
) -> int:
  """Ingest all chunks into a persistent Chroma collection.

  Returns the number of chunks ingested.

  `owner_user_id` and `visibility` are stored in metadata for future
  per-user and visibility-aware retrieval, but not yet enforced.
  """

  CHROMA_DIR.mkdir(parents=True, exist_ok=True)

  client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
  collection = client.get_or_create_collection(name=collection_name)

  chunks = build_chunks()
  if not chunks:
    return 0

  ids = [c.id for c in chunks]
  texts = [c.text for c in chunks]
  metadatas = [
    {
      "document_path": str(c.document_path),
      "source": c.document_path.name,
      "chunk_id": c.id,
      "owner_user_id": owner_user_id,
      "visibility": visibility,
    }
    for c in chunks
  ]

  # For simplicity we let Chroma handle embedding (default embedding function).
  collection.add(ids=ids, documents=texts, metadatas=metadatas)
  return len(chunks)


def ingest_files(
  paths: list[str],
  collection_name: str = "bsk_rag",
  owner_user_id: str = "admin",
  visibility: str = "private",
) -> int:
  """Ingest only the given file paths into the Chroma collection.

  - Skips non-existent files and non-.txt files.
  - Uses the same chunking + metadata schema as ingest_chunks.
  - Records `owner_user_id` and `visibility` in metadata for future
    per-user and sharing-aware retrieval.
  """

  CHROMA_DIR.mkdir(parents=True, exist_ok=True)

  client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
  collection = client.get_or_create_collection(name=collection_name)

  chunks: list[Chunk] = []

  for raw_path in paths:
    path = Path(raw_path)
    if not path.is_file() or path.suffix.lower() != ".txt":
      continue

    text = path.read_text(encoding="utf-8", errors="ignore")
    raw_chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    for idx, chunk_text in enumerate(raw_chunks):
      chunk_id = f"{path.name}::{idx}"
      chunks.append(Chunk(id=chunk_id, document_path=path, text=chunk_text))

  if not chunks:
    return 0

  ids = [c.id for c in chunks]
  texts = [c.text for c in chunks]
  metadatas = [
    {
      "document_path": str(c.document_path),
      "source": c.document_path.name,
      "chunk_id": c.id,
      "owner_user_id": owner_user_id,
      "visibility": visibility,
    }
    for c in chunks
  ]

  collection.add(ids=ids, documents=texts, metadatas=metadatas)
  return len(chunks)
