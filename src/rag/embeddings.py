"""Embedding model factory.

Dispatches on ``settings.embedding_provider``:
  * ``fastembed`` (default) — local ONNX model from ``fastembed``, zero-cost
  * ``openai`` — :class:`langchain_openai.OpenAIEmbeddings` (paid)

The result is cached process-wide so the model is loaded only once.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a process-wide cached embeddings client configured from settings."""
    settings = get_settings()

    if settings.embedding_provider == "fastembed":
        # Local ONNX inference. First call downloads the model (~30MB for
        # BAAI/bge-small-en-v1.5) and caches it under ~/.cache/fastembed/.
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=settings.embedding_model)

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if settings.openai_api_key is None:
            raise RuntimeError(
                "embedding_provider='openai' but OPENAI_API_KEY is not set."
            )
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    raise RuntimeError(f"unknown embedding_provider: {settings.embedding_provider!r}")
