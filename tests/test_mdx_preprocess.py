"""Tests for :func:`src.rag.ingestion.preprocess_mdx`.

Each test targets one representative MDX shape. The guiding invariant: anything
inside fenced code blocks or inline code spans must survive byte-for-byte, while
JSX/import noise in prose is removed.
"""

from __future__ import annotations

from src.rag.ingestion import preprocess_mdx


def test_removes_import_and_export_statements() -> None:
    src = (
        "import Tabs from '@theme/Tabs';\n"
        "import TabItem from '@theme/TabItem';\n"
        "export const meta = { title: 'x' };\n"
        "\n"
        "# Real Heading\n"
        "\n"
        "Body text.\n"
    )
    out = preprocess_mdx(src)
    assert "import" not in out
    assert "export" not in out
    assert "# Real Heading" in out
    assert "Body text." in out


def test_strips_paired_jsx_tags_but_keeps_inner_text() -> None:
    src = (
        "<Tabs>\n"
        '  <TabItem value="py" label="Python">\n'
        "  Install with pip.\n"
        "  </TabItem>\n"
        "</Tabs>\n"
    )
    out = preprocess_mdx(src)
    assert "Install with pip." in out
    assert "<Tabs>" not in out
    assert "TabItem" not in out


def test_strips_self_closing_jsx_tags() -> None:
    src = "Some text.\n\n<ImageZoom src='/img/a.png' alt='diagram' />\n\nMore text.\n"
    out = preprocess_mdx(src)
    assert "ImageZoom" not in out
    assert "Some text." in out
    assert "More text." in out


def test_preserves_fenced_code_block_verbatim() -> None:
    # The code block deliberately contains things the stripper would mangle in
    # prose: an import line, a JSX-looking tag, and a brace expression.
    src = (
        "Intro paragraph.\n"
        "\n"
        "```python\n"
        "import os  # this import MUST survive\n"
        "config = {'key': 'value'}  # braces survive\n"
        "print('<NotATag>')\n"
        "```\n"
        "\n"
        "Outro.\n"
    )
    out = preprocess_mdx(src)
    assert "import os  # this import MUST survive" in out
    assert "config = {'key': 'value'}  # braces survive" in out
    assert "print('<NotATag>')" in out
    assert "Intro paragraph." in out
    assert "Outro." in out


def test_preserves_inline_code_with_braces_and_tags() -> None:
    src = "Use `{variable}` syntax and the `<Component>` tag in LCEL.\n"
    out = preprocess_mdx(src)
    assert "`{variable}`" in out
    assert "`<Component>`" in out


def test_removes_bare_jsx_expression_in_prose() -> None:
    src = "The value is {someRuntimeVar} at render time.\n"
    out = preprocess_mdx(src)
    assert "{someRuntimeVar}" not in out
    assert "The value is" in out
    assert "at render time." in out


def test_removes_mdx_comment() -> None:
    src = "Before.\n\n{/* this is an MDX comment */}\n\nAfter.\n"
    out = preprocess_mdx(src)
    assert "MDX comment" not in out
    assert "Before." in out
    assert "After." in out


def test_pure_markdown_passes_through_essentially_unchanged() -> None:
    src = (
        "# Title\n"
        "\n"
        "A paragraph with a [link](https://example.com) and `inline code`.\n"
        "\n"
        "- bullet one\n"
        "- bullet two\n"
    )
    out = preprocess_mdx(src)
    assert "# Title" in out
    assert "[link](https://example.com)" in out
    assert "`inline code`" in out
    assert "- bullet one" in out


def test_markdown_autolink_is_not_treated_as_jsx() -> None:
    src = "See <https://python.langchain.com> for docs.\n"
    out = preprocess_mdx(src)
    assert "<https://python.langchain.com>" in out


def test_collapses_excess_blank_lines() -> None:
    src = "Para one.\n\n\n\n\nPara two.\n"
    out = preprocess_mdx(src)
    assert "\n\n\n" not in out
    assert "Para one." in out
    assert "Para two." in out


# --- Real-world Mintlify (langchain-ai/docs) component shapes ---


def test_strips_mintlify_tip_and_note_components() -> None:
    src = (
        "<Tip>\n"
        "    Runtime context refers to local context.\n"
        "</Tip>\n"
        "\n"
        "<Note>\n"
        "**Agent = Model + Harness**\n"
        "</Note>\n"
    )
    out = preprocess_mdx(src)
    assert "Tip" not in out
    assert "Note" not in out
    assert "Runtime context refers to local context." in out
    assert "Agent = Model + Harness" in out


def test_strips_multiline_self_closing_img_with_style_expression() -> None:
    src = (
        "Some prose.\n"
        "\n"
        "<img\n"
        '    src="/oss/images/core_agent_loop.png"\n'
        '    alt="Core agent loop diagram"\n'
        '    style={{height: "200px", width: "auto"}}\n'
        '    className="rounded-lg block mx-auto"\n'
        "/>\n"
        "\n"
        "More prose.\n"
    )
    out = preprocess_mdx(src)
    assert "img" not in out
    assert "core_agent_loop.png" not in out
    assert "style={{" not in out
    assert "Some prose." in out
    assert "More prose." in out


def test_strips_snippet_imports_but_keeps_code_block_imports() -> None:
    src = (
        "import AgentInvocationPy from '/snippets/code-samples/agent-invocation-py.mdx';\n"
        "\n"
        "An agent is a model calling tools in a loop.\n"
        "\n"
        "```typescript\n"
        'import { createAgent } from "langchain";\n'
        'const agent = createAgent({ model: "openai:gpt-5.4" });\n'
        "```\n"
    )
    out = preprocess_mdx(src)
    # The MDX import statement (prose) is gone...
    assert "AgentInvocationPy" not in out
    assert "/snippets/" not in out
    # ...but the import *inside* the code block survives verbatim.
    assert 'import { createAgent } from "langchain";' in out
    assert "An agent is a model calling tools in a loop." in out


def test_strips_mintlify_container_directives() -> None:
    src = (
        ":::python\n"
        "```python\n"
        "from langchain.agents import create_agent\n"
        "```\n"
        ":::\n"
        ":::js\n"
        "```typescript\n"
        'import { createAgent } from "langchain";\n'
        "```\n"
        ":::\n"
    )
    out = preprocess_mdx(src)
    assert ":::" not in out
    # Both code blocks are preserved.
    assert "from langchain.agents import create_agent" in out
    assert 'import { createAgent } from "langchain";' in out
