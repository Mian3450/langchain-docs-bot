"""Format answers + source citations for Telegram (HTML parse mode).

We use Telegram's HTML parse mode rather than MarkdownV2: the LLM answer text
needs only three characters escaped (``&``, ``<``, ``>``), whereas MarkdownV2
requires escaping ~18 characters, which mangles code-heavy answers.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.rag.generator import Source

SOURCES_HEADER = "\n\n📚 <b>Sources:</b>\n"


def escape_html(text: str) -> str:
    """Escape the characters Telegram's HTML parse mode treats specially."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_sources_block(sources: Iterable[Source]) -> str:
    """Render an ordered ``[N] title`` list with clickable GitHub links.

    Returns an empty string when there are no sources.
    """
    sources = list(sources)
    if not sources:
        return ""
    lines = [SOURCES_HEADER.rstrip("\n")]
    for s in sources:
        title = escape_html(s.title)
        if s.source_url:
            lines.append(f'[{s.id}] <a href="{escape_html(s.source_url)}">{title}</a>')
        else:
            lines.append(f"[{s.id}] {title}")
    return "\n".join(lines)


def format_reply(answer_text: str, sources: Iterable[Source]) -> str:
    """Combine the (HTML-escaped) answer with the sources block."""
    return escape_html(answer_text) + format_sources_block(sources)


def split_for_telegram(text: str, max_length: int) -> list[str]:
    """Split ``text`` into chunks no longer than ``max_length`` for Telegram.

    Splits on paragraph, then line, then hard character boundaries so we never
    exceed the limit. HTML tags in this project are short and self-contained per
    line (links), so line-boundary splitting won't break a tag across messages.
    """
    if len(text) <= max_length:
        return [text]

    parts: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        window = remaining[:max_length]
        # Prefer a paragraph break, then a newline, then a space.
        split_at = window.rfind("\n\n")
        if split_at == -1:
            split_at = window.rfind("\n")
        if split_at == -1:
            split_at = window.rfind(" ")
        if split_at == -1:
            split_at = max_length  # no natural boundary; hard cut
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts
