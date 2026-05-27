"""Answer generation with numeric-ID source citations.

Each retrieved chunk is assigned a short numeric ID (``[1]``, ``[2]``, ...)
before being placed in the prompt. The LLM cites by ID; the bot layer later maps
those IDs back to clickable GitHub URLs (see ``src/bot/handlers.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_settings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Source:
    """A citable source backing an answer. ``id`` matches the ``[N]`` in text."""

    id: int
    title: str
    source_path: str
    source_url: str


@dataclass
class AnswerResult:
    """Result of the RAG pipeline returned to the bot layer."""

    text: str  # Raw LLM output, retains inline [N] citation markers.
    sources: list[Source] = field(default_factory=list)
    latency_ms: float = 0.0


_SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in the LangChain framework.\n"
    "Answer the user's question using ONLY the provided context.\n"
    "If the context doesn't contain the answer, say so honestly — do not invent facts.\n"
    "Cite the sources you used by their numeric ID, like [1] or [2, 3].\n"
    "Place citations at the end of sentences they support."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)

_NO_CONTEXT_MESSAGE = (
    "I couldn't find anything in the LangChain documentation to answer that. "
    "Try rephrasing, or ask about a specific LangChain concept (chains, agents, "
    "retrievers, LCEL, callbacks, ...)."
)


def _build_context(docs: list[Document]) -> tuple[str, list[Source]]:
    """Format retrieved chunks into a numbered context block + Source list."""
    blocks: list[str] = []
    sources: list[Source] = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "untitled")
        blocks.append(f"[{i}] (from {title})\n{doc.page_content}")
        sources.append(
            Source(
                id=i,
                title=str(title),
                source_path=str(doc.metadata.get("source_path", "")),
                source_url=str(doc.metadata.get("source_url", "")),
            )
        )
    return "\n\n".join(blocks), sources


def _build_llm() -> BaseChatModel:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )


async def generate_answer(
    question: str,
    docs: list[Document],
    *,
    llm: BaseChatModel | None = None,
) -> AnswerResult:
    """Generate an answer grounded in ``docs`` with ``[N]`` citations.

    Args:
        question: The user's question.
        docs: Retrieved chunks, best-first. IDs are assigned by position.
        llm: Optional chat model override (used by tests to inject a mock).

    Returns:
        An :class:`AnswerResult`. If ``docs`` is empty, returns a canned
        "no context" answer without calling the LLM (saves tokens).
    """
    if not docs:
        return AnswerResult(text=_NO_CONTEXT_MESSAGE, sources=[], latency_ms=0.0)

    context, sources = _build_context(docs)
    model = llm if llm is not None else _build_llm()
    chain = _PROMPT | model

    start = time.perf_counter()
    response = await chain.ainvoke({"context": context, "question": question})
    latency_ms = (time.perf_counter() - start) * 1000.0

    text = response.content if isinstance(response.content, str) else str(response.content)
    log.info("answer_generated", latency_ms=round(latency_ms, 1), n_sources=len(sources))
    return AnswerResult(text=text.strip(), sources=sources, latency_ms=latency_ms)
