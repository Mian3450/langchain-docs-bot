"""Tests for chunking behaviour in src.rag.ingestion."""

from __future__ import annotations

from langchain_core.documents import Document

from src.rag.ingestion import (
    build_source_url,
    chunk_documents,
    extract_section,
    extract_title,
)


def _make_doc(text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source_path": "src/oss/concepts/x.mdx",
            "source_url": "https://github.com/langchain-ai/docs/blob/main/src/oss/concepts/x.mdx",
            "title": "X",
            "section": "concepts",
        },
    )


def test_chunks_respect_max_size() -> None:
    long_text = "word " * 2000  # ~10k chars
    chunks = chunk_documents([_make_doc(long_text)], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    # Allow a small slack: the splitter may exceed slightly on hard boundaries.
    assert all(len(c.page_content) <= 500 + 50 for c in chunks)


def test_chunk_metadata_is_propagated() -> None:
    chunks = chunk_documents([_make_doc("word " * 2000)], chunk_size=400, chunk_overlap=40)
    for c in chunks:
        assert c.metadata["source_url"].endswith("src/oss/concepts/x.mdx")
        assert c.metadata["title"] == "X"
        assert c.metadata["section"] == "concepts"


def test_short_fragments_are_filtered_out() -> None:
    docs = [_make_doc("tiny"), _make_doc("word " * 500)]
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=30, min_chunk_chars=100)
    assert all(len(c.page_content.strip()) >= 100 for c in chunks)


def test_overlap_creates_shared_content() -> None:
    # Distinct tokens so we can detect overlap between consecutive chunks.
    text = " ".join(f"tok{i}" for i in range(1000))
    chunks = chunk_documents([_make_doc(text)], chunk_size=200, chunk_overlap=80)
    assert len(chunks) >= 2
    # Some token from the tail of chunk[0] should reappear at the head of chunk[1].
    first_tail = set(chunks[0].page_content.split()[-10:])
    second_head = set(chunks[1].page_content.split()[:20])
    assert first_tail & second_head


def test_extract_section_fallback_to_misc() -> None:
    assert extract_section("src/oss/concepts/context.mdx") == "concepts"
    assert extract_section("src/oss/langchain/agents.mdx") == "langchain"
    assert extract_section("src/oss/common-errors.mdx") == "misc"
    # Custom subdir is honoured.
    assert extract_section("docs/docs/concepts/x.mdx", "docs/docs") == "concepts"


def test_extract_title_precedence() -> None:
    assert extract_title("# Body Heading\n", {"title": "FM Title"}, "f.mdx") == "FM Title"
    assert extract_title("# Body Heading\n", {}, "f.mdx") == "Body Heading"
    assert extract_title("no heading here", {}, "my_file.mdx") == "My File"


def test_build_source_url() -> None:
    url = build_source_url("src/oss/concepts/context.mdx", "langchain-ai/docs", "main")
    assert url == (
        "https://github.com/langchain-ai/docs/blob/main/src/oss/concepts/context.mdx"
    )
