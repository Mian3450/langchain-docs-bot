"""Document ingestion: fetch LangChain docs, clean MDX, chunk, attach metadata.

The pipeline has two clearly separated concerns:

1. :func:`preprocess_mdx` — a pure, well-tested function that turns raw ``.mdx``
   (Markdown + JSX) into clean Markdown. This is the trickiest unit, so it is
   isolated and covered by ``tests/test_mdx_preprocess.py``.
2. The scraping + chunking orchestration (:func:`load_documents`,
   :func:`chunk_documents`, :func:`build_documents`) that produces the
   ``langchain_core.documents.Document`` objects fed into the vector store.
"""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter
import httpx
import structlog
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = structlog.get_logger(__name__)

# Default path within the docs repo that holds the LangChain OSS docs sources.
# (LangChain moved its docs out of the monorepo into langchain-ai/docs; the
# open-source docs now live under src/oss. Override via Settings.docs_subdir.)
DEFAULT_DOCS_SUBDIR = "src/oss"
# File extensions we treat as documentation.
DOC_EXTENSIONS = (".md", ".mdx")
# Chunks shorter than this are dropped as nav/boilerplate fragments.
MIN_CHUNK_CHARS = 100
# Markdown-aware split boundaries, coarsest first.
MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]

# ---------------------------------------------------------------------------
# MDX preprocessing
# ---------------------------------------------------------------------------
#
# Strategy (per PROJECT.md): NEVER run the stripping regexes over the whole
# document — fenced code blocks and inline code must survive untouched. So we
# split the text into code / non-code segments first, transform only the
# non-code ("prose") segments, then rejoin.

# A fenced code block: an opening fence (3+ backticks or tildes) on its own
# line, anything until a closing fence of the *same character run*, matched
# non-greedily so adjacent blocks don't merge.
_FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ \t]*(?P=fence)[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

# Inline code span — protected inside prose so brace/JSX stripping can't touch
# example snippets like `{foo}` or `<Tag>`.
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# `import X from 'y';` / `import './side-effect';` — whole line.
_IMPORT_RE = re.compile(r"^[ \t]*import\s+[^\n]*$", re.MULTILINE)
# `export const X = ...` / `export default ...` — whole line.
_EXPORT_RE = re.compile(r"^[ \t]*export\s+[^\n]*$", re.MULTILINE)

# MDX comments: `{/* ... */}`, possibly spanning lines.
_MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)

# HTML/JSX tags: opening, closing, or self-closing. Restricted to names that
# start with a letter so Markdown autolinks (`<https://...>`, `<a@b.com>`) and
# generic angle-bracket prose are left alone.
_JSX_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9.]*(?:\s[^<>]*?)?/?>")

# A standalone JSX expression in prose, e.g. `{someVariable}`. Single line, no
# nested braces — deliberately conservative.
_JSX_EXPR_RE = re.compile(r"\{[^{}\n]*\}")

# Mintlify/Docusaurus container directive markers on their own line, e.g.
# `:::python`, `:::js`, `:::note Title`, or the closing `:::`. We drop the
# marker lines and keep the wrapped content.
_CONTAINER_DIRECTIVE_RE = re.compile(r"^[ \t]*:::.*$", re.MULTILINE)

# 3+ consecutive newlines collapse to a single blank line.
_BLANKS_RE = re.compile(r"\n{3,}")


def _strip_prose(segment: str) -> str:
    """Apply MDX stripping to a non-code segment (inline code protected)."""
    # Protect inline code spans with placeholders unlikely to occur in docs.
    spans: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"\x00INLINE{len(spans) - 1}\x00"

    segment = _INLINE_CODE_RE.sub(_stash, segment)

    segment = _MDX_COMMENT_RE.sub("", segment)
    segment = _IMPORT_RE.sub("", segment)
    segment = _EXPORT_RE.sub("", segment)
    segment = _CONTAINER_DIRECTIVE_RE.sub("", segment)
    # Drop JSX tags but keep the inner text of paired components.
    segment = _JSX_TAG_RE.sub("", segment)
    segment = _JSX_EXPR_RE.sub("", segment)

    # Restore inline code.
    def _restore(match: re.Match[str]) -> str:
        return spans[int(match.group(1))]

    segment = re.sub(r"\x00INLINE(\d+)\x00", _restore, segment)
    return segment


def preprocess_mdx(content: str) -> str:
    """Strip MDX-specific syntax, leaving clean Markdown.

    Removes ``import``/``export`` statements, JSX component tags (keeping the
    inner text of paired components), MDX comments, and bare JSX expressions.
    Fenced code blocks and inline code spans are preserved verbatim.

    Args:
        content: Raw ``.mdx`` (or ``.md``) file contents.

    Returns:
        Cleaned Markdown with at most one consecutive blank line between blocks
        and no leading/trailing whitespace.
    """
    out: list[str] = []
    last = 0
    for match in _FENCE_RE.finditer(content):
        if match.start() > last:
            out.append(_strip_prose(content[last : match.start()]))
        out.append(match.group(0))  # code block, verbatim
        last = match.end()
    if last < len(content):
        out.append(_strip_prose(content[last:]))

    cleaned = "".join(out)
    cleaned = _BLANKS_RE.sub("\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

_FIRST_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def extract_section(repo_rel_path: str, docs_subdir: str = DEFAULT_DOCS_SUBDIR) -> str:
    """Return the top-level docs section for a repo-relative path.

    ``src/oss/concepts/context.mdx`` -> ``"concepts"``. Files sitting directly in
    ``docs_subdir`` (or anything unexpected) fall back to ``"misc"`` so callers
    never have to handle a missing section.
    """
    prefix = docs_subdir.rstrip("/") + "/"
    rel = repo_rel_path[len(prefix) :] if repo_rel_path.startswith(prefix) else repo_rel_path
    parts = rel.split("/")
    return parts[0] if len(parts) > 1 and parts[0] else "misc"


def extract_title(body: str, fm_meta: dict[str, object], filename: str) -> str:
    """Pick a human title: frontmatter ``title`` > first ``# H1`` > filename."""
    fm_title = fm_meta.get("title")
    if isinstance(fm_title, str) and fm_title.strip():
        return fm_title.strip()
    h1 = _FIRST_H1_RE.search(body)
    if h1:
        return h1.group(1).strip()
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def build_source_url(repo_rel_path: str, repo: str, ref: str) -> str:
    """Construct the GitHub blob URL for a repo-relative documentation path."""
    return f"https://github.com/{repo}/blob/{ref}/{repo_rel_path}"


# ---------------------------------------------------------------------------
# Fetching (GitHub) — runs offline as part of scripts/ingest_docs.py
# ---------------------------------------------------------------------------


def fetch_docs_to_disk(
    raw_dir: str | Path,
    repo: str,
    ref: str,
    *,
    docs_subdir: str = DEFAULT_DOCS_SUBDIR,
    limit: int | None = None,
    github_token: str | None = None,
) -> list[Path]:
    """Download all docs ``.md``/``.mdx`` files from the repo into ``raw_dir``.

    Uses the GitHub Trees API (one request) to enumerate paths, then fetches raw
    file contents from ``raw.githubusercontent.com`` (which is not subject to
    the API rate limit). Files are written mirroring their repo-relative path.

    Args:
        raw_dir: Local destination directory.
        repo: ``owner/name`` slug, e.g. ``"langchain-ai/docs"``.
        ref: Git ref (branch/tag/sha).
        docs_subdir: Repo path prefix to restrict to, e.g. ``"src/oss"``.
        limit: If set, fetch at most this many files (useful for smoke runs).
        github_token: Optional token to raise the API rate limit.

    Returns:
        Sorted list of local file paths that were written.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    prefix = docs_subdir.rstrip("/") + "/"

    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
    log.info("fetching_docs_tree", repo=repo, ref=ref, subdir=docs_subdir)
    with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
        resp = client.get(tree_url)
        resp.raise_for_status()
        tree = resp.json()
        if tree.get("truncated"):
            log.warning("github_tree_truncated", note="some paths may be missing")

        doc_paths = [
            item["path"]
            for item in tree.get("tree", [])
            if item["type"] == "blob"
            and item["path"].startswith(prefix)
            and item["path"].endswith(DOC_EXTENSIONS)
        ]
        doc_paths.sort()
        if limit is not None:
            doc_paths = doc_paths[:limit]
        log.info("downloading_docs", count=len(doc_paths))

        written: list[Path] = []
        for repo_rel_path in doc_paths:
            raw_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{repo_rel_path}"
            r = client.get(raw_url)
            if r.status_code != 200:
                log.warning("download_failed", path=repo_rel_path, status=r.status_code)
                continue
            dest = raw_path / repo_rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(r.text, encoding="utf-8")
            written.append(dest)

    log.info("download_complete", written=len(written), dest=str(raw_path))
    return sorted(written)


# ---------------------------------------------------------------------------
# Loading + chunking
# ---------------------------------------------------------------------------


def _repo_rel_path(file_path: Path, raw_dir: Path, docs_subdir: str) -> str:
    """Recover the repo-relative path from a file under ``raw_dir``.

    Files are stored mirroring their repo path; if the ``docs_subdir`` segment is
    present we anchor on it, otherwise we use the path relative to ``raw_dir``.
    """
    prefix = docs_subdir.rstrip("/") + "/"
    posix = file_path.relative_to(raw_dir).as_posix()
    if posix.startswith(prefix):
        return posix
    return f"{prefix}{posix}"


def load_documents(
    raw_dir: str | Path,
    repo: str,
    ref: str,
    docs_subdir: str = DEFAULT_DOCS_SUBDIR,
) -> list[Document]:
    """Load + clean every doc file under ``raw_dir`` into LangChain Documents.

    Each document carries metadata: ``source_path`` (repo-relative),
    ``source_url`` (GitHub blob URL), ``title``, and ``section``.
    """
    raw_path = Path(raw_dir)
    files = sorted(
        p for p in raw_path.rglob("*") if p.is_file() and p.suffix.lower() in DOC_EXTENSIONS
    )
    log.info("loading_documents", count=len(files), raw_dir=str(raw_path))

    documents: list[Document] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        post = frontmatter.loads(text)
        body = preprocess_mdx(post.content)
        if not body.strip():
            continue

        repo_rel = _repo_rel_path(file_path, raw_path, docs_subdir)
        metadata = {
            "source_path": repo_rel,
            "source_url": build_source_url(repo_rel, repo, ref),
            "title": extract_title(body, post.metadata, file_path.name),
            "section": extract_section(repo_rel, docs_subdir),
        }
        documents.append(Document(page_content=body, metadata=metadata))

    log.info("documents_loaded", count=len(documents))
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    *,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> list[Document]:
    """Split documents into chunks and drop short boilerplate fragments.

    Uses :class:`RecursiveCharacterTextSplitter` with Markdown-aware separators.
    Source metadata is propagated to every chunk by the splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=MARKDOWN_SEPARATORS,
        keep_separator=True,
    )
    chunks = splitter.split_documents(documents)
    kept = [c for c in chunks if len(c.page_content.strip()) >= min_chunk_chars]
    log.info(
        "chunking_complete",
        input_docs=len(documents),
        raw_chunks=len(chunks),
        kept_chunks=len(kept),
        dropped=len(chunks) - len(kept),
    )
    return kept
