"""Application settings, loaded from environment / `.env` via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed configuration for the bot and RAG pipeline.

    Values are read from environment variables (case-insensitive) or a local
    `.env` file. Secrets use :class:`~pydantic.SecretStr` so they are never
    accidentally logged or serialised.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Required secrets ---
    telegram_bot_token: SecretStr
    openai_api_key: SecretStr

    # --- Model configuration ---
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2

    # --- RAG parameters ---
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=5, gt=0)
    # "similarity" or "mmr" — see src/rag/retriever.py
    retrieval_strategy: Literal["similarity", "mmr"] = "similarity"

    # --- Paths ---
    chroma_persist_dir: str = "./data/chroma"
    raw_docs_dir: str = "./data/raw"
    # Chroma collection name; bump when the embedding model/chunking changes.
    collection_name: str = "langchain_docs"

    # --- Bot behavior ---
    max_message_length: int = Field(default=4000, gt=0, le=4096)
    rate_limit_per_minute: int = Field(default=10, gt=0)

    # --- Logging ---
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    # --- Documentation source (GitHub) ---
    # LangChain moved its docs out of the monorepo into langchain-ai/docs; the
    # open-source docs live under src/oss on the `main` branch.
    docs_repo: str = "langchain-ai/docs"
    docs_ref: str = "main"
    docs_subdir: str = "src/oss"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Cached so configuration is parsed once. Call ``get_settings.cache_clear()``
    in tests that need to re-read the environment.
    """
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
