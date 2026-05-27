"""Shared pytest fixtures and test configuration.

All tests run fully offline: dummy API keys are injected into the environment so
``get_settings()`` succeeds, and no test makes a real network/LLM/embeddings
call (the LLM is injected as a fake; retrieval uses a deterministic local
embedding).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Inject dummy secrets before any `src` import triggers settings parsing.
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Ensure each test reads fresh settings (the factory is lru_cached)."""
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
