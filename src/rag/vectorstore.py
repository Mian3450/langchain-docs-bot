"""ChromaDB vector store wrapper.

Provides a cached :class:`~langchain_chroma.Chroma` instance backed by a
persistent on-disk collection, plus helpers used by the ingestion script to
(re)build the collection.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import structlog
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_settings
from src.rag.embeddings import get_embeddings

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Return a process-wide cached, persistent Chroma vector store.

    The collection is created lazily on first write if it does not yet exist.
    The directory is created if missing so a fresh checkout works out of the box.
    """
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def add_documents(chunks: list[Document], *, batch_size: int = 256) -> int:
    """Embed and persist chunks into the collection, in batches.

    Returns the number of chunks added. Batching keeps embedding requests and
    Chroma writes to a reasonable size for large corpora.
    """
    store = get_vectorstore()
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        store.add_documents(batch)
        total += len(batch)
        log.info("chunks_persisted", added=total, total=len(chunks))
    return total


def reset_collection() -> None:
    """Delete all vectors in the configured collection (for clean re-ingestion)."""
    store = get_vectorstore()
    store.reset_collection()
    log.info("collection_reset", collection=get_settings().collection_name)


def count() -> int:
    """Return the number of stored vectors in the collection."""
    return get_vectorstore()._collection.count()
