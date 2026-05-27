"""Retrieval over the Chroma vector store.

Supports two strategies:

* ``"similarity"`` (default) — plain top-k cosine similarity.
* ``"mmr"`` — Maximal Marginal Relevance, which trades a little relevance for
  diversity so the context isn't five near-duplicate chunks.

Both return :class:`~langchain_core.documents.Document` objects carrying the
ingestion metadata (``source_url``, ``title``, ``section``) used for citations.
"""

from __future__ import annotations

import structlog
from langchain_core.documents import Document

from src.config import get_settings
from src.rag.vectorstore import get_vectorstore

log = structlog.get_logger(__name__)

# For MMR: how many candidates to fetch before re-ranking down to top_k.
_MMR_FETCH_MULTIPLIER = 4


def retrieve(
    question: str,
    *,
    top_k: int | None = None,
    strategy: str | None = None,
) -> list[Document]:
    """Retrieve the most relevant document chunks for a question.

    Args:
        question: The user's natural-language query.
        top_k: Number of chunks to return. Defaults to ``settings.top_k``.
        strategy: ``"similarity"`` or ``"mmr"``. Defaults to
            ``settings.retrieval_strategy``.

    Returns:
        A list of up to ``top_k`` chunks ordered best-first.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k
    strat = strategy or settings.retrieval_strategy
    store = get_vectorstore()

    if strat == "mmr":
        docs = store.max_marginal_relevance_search(
            question, k=k, fetch_k=k * _MMR_FETCH_MULTIPLIER
        )
    else:
        docs = store.similarity_search(question, k=k)

    log.info("retrieved", strategy=strat, k=k, returned=len(docs))
    return docs
