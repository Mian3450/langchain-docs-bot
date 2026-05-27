"""Embedding model factory.

Wraps :class:`langchain_openai.OpenAIEmbeddings` behind a cached factory so the
client (and its config) is created once per process.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from src.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Return a process-wide cached embeddings client configured from settings."""
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
