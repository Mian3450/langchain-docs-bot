"""Run-once ingestion: fetch LangChain docs, chunk, embed, and persist to Chroma.

Usage:
    python -m scripts.ingest_docs                 # full fetch + ingest
    python -m scripts.ingest_docs --limit 10      # quick smoke run (10 files)
    python -m scripts.ingest_docs --reset         # wipe collection first
    python -m scripts.ingest_docs --skip-fetch    # reuse already-downloaded raw

Requires OPENAI_API_KEY (embeddings cost a few cents for the full corpus).
A GITHUB_TOKEN env var is optional and only raises the GitHub API rate limit.
"""

from __future__ import annotations

import argparse
import os

import structlog

from src.config import get_settings
from src.rag.ingestion import chunk_documents, fetch_docs_to_disk, load_documents
from src.rag.vectorstore import add_documents, count, reset_collection
from src.utils.logging import configure_logging
from src.utils.tls import enable_system_trust_store

log = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest LangChain docs into ChromaDB.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Fetch at most N files (smoke runs)."
    )
    parser.add_argument(
        "--reset", action="store_true", help="Clear the collection before ingesting."
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip downloading; reuse files already in the raw docs dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    enable_system_trust_store()

    if not args.skip_fetch:
        fetch_docs_to_disk(
            settings.raw_docs_dir,
            repo=settings.docs_repo,
            ref=settings.docs_ref,
            docs_subdir=settings.docs_subdir,
            limit=args.limit,
            github_token=os.getenv("GITHUB_TOKEN"),
        )

    documents = load_documents(
        settings.raw_docs_dir, settings.docs_repo, settings.docs_ref, settings.docs_subdir
    )
    if not documents:
        log.error("no_documents_found", hint="run without --skip-fetch first")
        raise SystemExit(1)

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)

    if args.reset:
        reset_collection()

    added = add_documents(chunks)
    log.info("ingestion_done", chunks_added=added, total_vectors=count())


if __name__ == "__main__":
    main()
