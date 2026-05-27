"""Tests for src.rag.retriever.

Uses a deterministic bag-of-words embedding + a real in-memory Chroma store, so
retrieval is exercised end-to-end (no OpenAI calls). A known question should
surface the topically matching document.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import src.rag.retriever as retriever_mod
from src.rag.retriever import retrieve

_VOCAB_DIM = 256


class DeterministicBagOfWords(Embeddings):
    """Hash tokens into a fixed-dim, L2-normalised vector. Pure & offline."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * _VOCAB_DIM
        for tok in text.lower().split():
            v[hash(tok) % _VOCAB_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


_DOCS = [
    Document(
        page_content="An agent uses an LLM to decide which tools to call and in what order.",
        metadata={"title": "Agents", "source_url": "url/agents", "section": "concepts"},
    ),
    Document(
        page_content="A retriever returns relevant documents for a query string.",
        metadata={"title": "Retrievers", "source_url": "url/retrievers", "section": "concepts"},
    ),
    Document(
        page_content="LCEL composes runnables with the pipe operator for streaming and async.",
        metadata={"title": "LCEL", "source_url": "url/lcel", "section": "concepts"},
    ),
]


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> Iterator[Chroma]:
    # Unique collection per test so in-process Chroma state can't leak between
    # tests (the default client is shared within the process).
    store = Chroma(
        collection_name=f"test_docs_{uuid.uuid4().hex}",
        embedding_function=DeterministicBagOfWords(),
    )
    store.add_documents(_DOCS)
    # Patch the cached factory used by retrieve().
    monkeypatch.setattr(retriever_mod, "get_vectorstore", lambda: store)
    yield store
    store.delete_collection()


def test_similarity_retrieves_topical_document(fake_store: Chroma) -> None:
    docs = retrieve("which tools should the agent call", top_k=1, strategy="similarity")
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Agents"


def test_top_k_limits_results(fake_store: Chroma) -> None:
    docs = retrieve("agent retriever lcel", top_k=2, strategy="similarity")
    assert len(docs) == 2


def test_mmr_strategy_returns_results(fake_store: Chroma) -> None:
    docs = retrieve("how do I retrieve relevant documents", top_k=2, strategy="mmr")
    assert 1 <= len(docs) <= 2
    # Metadata (used for citations) must survive retrieval.
    assert all("source_url" in d.metadata for d in docs)


def test_defaults_come_from_settings(fake_store: Chroma) -> None:
    # No explicit top_k/strategy -> uses settings defaults (top_k=5, similarity).
    docs = retrieve("agent")
    assert len(docs) == len(_DOCS)  # fewer than top_k=5 available, returns all
