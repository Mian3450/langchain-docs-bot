"""Tests for src.utils.source_formatter (Telegram reply rendering)."""

from __future__ import annotations

from src.rag.generator import Source
from src.utils.source_formatter import (
    escape_html,
    format_reply,
    format_sources_block,
    split_for_telegram,
)


def _sources() -> list[Source]:
    return [
        Source(1, "LCEL", "docs/docs/concepts/lcel.mdx", "https://github.com/x/lcel.mdx"),
        Source(2, "Streaming", "docs/docs/how_to/stream.mdx", "https://github.com/x/stream.mdx"),
    ]


def test_escape_html_escapes_special_chars() -> None:
    assert escape_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_format_sources_block_renders_numbered_links() -> None:
    block = format_sources_block(_sources())
    assert "📚" in block
    assert '[1] <a href="https://github.com/x/lcel.mdx">LCEL</a>' in block
    assert '[2] <a href="https://github.com/x/stream.mdx">Streaming</a>' in block


def test_format_sources_block_empty_when_no_sources() -> None:
    assert format_sources_block([]) == ""


def test_format_reply_escapes_answer_but_keeps_citation_markers() -> None:
    reply = format_reply("Use <Tag> and a & b [1].", _sources())
    assert "Use &lt;Tag&gt; and a &amp; b [1]." in reply
    assert "📚" in reply


def test_split_short_text_returns_single_part() -> None:
    assert split_for_telegram("hello", 4000) == ["hello"]


def test_split_long_text_respects_max_length() -> None:
    text = "\n\n".join(f"Paragraph number {i} with some content." for i in range(500))
    parts = split_for_telegram(text, 200)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)
    # No content lost (modulo whitespace trimming at boundaries).
    assert "Paragraph number 0" in parts[0]
    assert "Paragraph number 499" in parts[-1]


def test_split_hard_cuts_when_no_boundary() -> None:
    text = "x" * 1000  # no spaces/newlines
    parts = split_for_telegram(text, 100)
    assert all(len(p) <= 100 for p in parts)
    assert sum(len(p) for p in parts) == 1000
