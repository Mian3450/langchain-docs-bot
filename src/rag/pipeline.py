"""End-to-end RAG orchestration.

:func:`answer_question` is the single entry point the bot handler calls:
retrieve relevant chunks, then generate a cited answer. Keeping this the only
public surface means the bot layer never needs to know about the vector store,
retrieval strategy, or prompt details.
"""

from __future__ import annotations

import structlog

from src.rag.generator import AnswerResult, generate_answer
from src.rag.retriever import retrieve

log = structlog.get_logger(__name__)


async def answer_question(question: str) -> AnswerResult:
    """Answer a question about LangChain using the RAG pipeline.

    Args:
        question: The user's natural-language question.

    Returns:
        An :class:`~src.rag.generator.AnswerResult` with ``.text`` (answer with
        inline ``[N]`` citations), ``.sources`` (ordered, indices match the
        citation IDs), and ``.latency_ms`` (generation latency).
    """
    question = question.strip()
    log.info("answering", question_len=len(question))
    docs = retrieve(question)
    result = await generate_answer(question, docs)
    return result
