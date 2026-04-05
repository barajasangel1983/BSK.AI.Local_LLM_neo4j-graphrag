"""Configuration helpers for the RAG lab.

For now we keep it very simple and file-based. Later we can move this to
proper settings/env variables.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Directory where raw documents will be stored for ingestion.
DATA_DIR = BASE_DIR / "data" / "raw"

# Directory where Chroma will persist its index.
CHROMA_DIR = BASE_DIR / "data" / "chroma_index"

# Default chunking parameters.
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 100  # characters

# Default number of results to retrieve.
DEFAULT_TOP_K = 5
