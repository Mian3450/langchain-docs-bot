"""Tests for src.rag.generator.

Uses a fake chat model that records the prompt it receives and returns a fixed
answer, so we can assert on prompt structure and source extraction without any
OpenAI calls.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.rag.generator import AnswerResult, Source, generate_answer


class RecordingChatModel(BaseChatModel):
    """Fake chat model: records received messages, returns a canned answer."""

    response: str = "LCEL is great [1]. It supports streaming [2]."
    sink: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.sink.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])

    @property
    def _llm_type(self) -> str:
        return "recording-fake"


def _docs() -> list[Document]:
    return [
        Document(
            page_content="LCEL composes runnables.",
            metadata={
                "title": "LCEL",
                "source_path": "docs/docs/concepts/lcel.mdx",
                "source_url": "https://github.com/x/blob/master/docs/docs/concepts/lcel.mdx",
            },
        ),
        Document(
            page_content="Streaming yields tokens incrementally.",
            metadata={
                "title": "Streaming",
                "source_path": "docs/docs/how_to/stream.mdx",
                "source_url": "https://github.com/x/blob/master/docs/docs/how_to/stream.mdx",
            },
        ),
    ]


async def test_generate_returns_answer_and_sources() -> None:
    llm = RecordingChatModel()
    result = await generate_answer("What is LCEL?", _docs(), llm=llm)

    assert isinstance(result, AnswerResult)
    assert result.text == "LCEL is great [1]. It supports streaming [2]."
    assert len(result.sources) == 2
    assert result.sources[0] == Source(
        id=1,
        title="LCEL",
        source_path="docs/docs/concepts/lcel.mdx",
        source_url="https://github.com/x/blob/master/docs/docs/concepts/lcel.mdx",
    )
    assert result.latency_ms >= 0.0


async def test_prompt_contains_numbered_context_and_question() -> None:
    llm = RecordingChatModel()
    await generate_answer("What is LCEL?", _docs(), llm=llm)

    # One invocation recorded; flatten its messages to text.
    assert len(llm.sink) == 1
    rendered = "\n".join(str(m.content) for m in llm.sink[0])
    assert "[1] (from LCEL)" in rendered
    assert "[2] (from Streaming)" in rendered
    assert "LCEL composes runnables." in rendered
    assert "What is LCEL?" in rendered
    # System instruction about citing by numeric ID is present.
    assert "numeric ID" in rendered


async def test_empty_context_short_circuits_without_calling_llm() -> None:
    llm = RecordingChatModel()
    result = await generate_answer("anything", [], llm=llm)

    assert llm.sink == []  # LLM never called
    assert result.sources == []
    assert "couldn't find" in result.text.lower()


@pytest.mark.parametrize("n", [1, 3, 5])
async def test_source_ids_are_sequential(n: int) -> None:
    docs = _docs()[:1] * n  # repeat to get n docs
    result = await generate_answer("q", docs, llm=RecordingChatModel())
    assert [s.id for s in result.sources] == list(range(1, n + 1))
