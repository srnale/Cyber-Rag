"""Configuration for the CyberRAG app."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_env_int(name: str, default: int) -> int:
    """Read an integer env var and coerce it to the provided default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    """Read a float env var and coerce it to the provided default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "allam-2-7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = _get_env_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _get_env_int("CHUNK_OVERLAP", 120)
TOP_K = _get_env_int("TOP_K", 5)
USE_HYBRID = os.getenv("USE_HYBRID", "false").strip().lower() == "true"
RELEVANCE_THRESHOLD = _get_env_float("RELEVANCE_THRESHOLD", 0.30)
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "cyber_docs"

# The model name can be changed through .env if the provider changes availability.
