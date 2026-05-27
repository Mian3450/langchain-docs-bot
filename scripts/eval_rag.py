"""Evaluate the RAG pipeline with DeepEval and write a Markdown report.

IMPORTANT: this is NOT a unit test. DeepEval uses an LLM as a judge, so:
  * it requires OPENAI_API_KEY and consumes tokens (~$0.10-0.50 for a full run),
  * scores are non-deterministic (they vary a few percent between runs),
  * it is slow (~1-2 min per question).

Run it manually before releases or after prompt changes:
    python -m scripts.eval_rag                 # full dataset
    python -m scripts.eval_rag --smoke         # 3 questions, quick sanity check

Install the eval extra first:  uv pip install -e ".[eval]"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from src.config import get_settings
from src.rag.generator import generate_answer
from src.rag.retriever import retrieve
from src.utils.logging import configure_logging
from src.utils.tls import enable_system_trust_store

log = structlog.get_logger(__name__)

DATASET_PATH = Path("tests/eval_dataset.json")
REPORT_PATH = Path("eval_report.md")
SMOKE_COUNT = 3


def _load_dataset(smoke: bool) -> list[dict[str, str]]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    items: list[dict[str, str]] = data["items"]
    return items[:SMOKE_COUNT] if smoke else items


async def _run_pipeline(question: str) -> tuple[str, list[str]]:
    """Return (answer_text, retrieval_context) for a single question."""
    docs = retrieve(question)
    result = await generate_answer(question, docs)
    return result.text, [d.page_content for d in docs]


def _build_metrics() -> list[Any]:
    # Imported lazily so the rest of the project doesn't depend on deepeval.
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        FaithfulnessMetric,
    )

    threshold = 0.7
    return [
        AnswerRelevancyMetric(threshold=threshold),
        FaithfulnessMetric(threshold=threshold),
        ContextualPrecisionMetric(threshold=threshold),
    ]


def _evaluate(items: list[dict[str, str]]) -> dict[str, list[float]]:
    from deepeval.test_case import LLMTestCase

    metrics = _build_metrics()
    scores: dict[str, list[float]] = {m.__class__.__name__: [] for m in metrics}

    for i, item in enumerate(items, start=1):
        question = item["question"]
        log.info("evaluating", index=i, total=len(items), topic=item.get("topic"))
        answer, context = asyncio.run(_run_pipeline(question))
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=item.get("expected_answer", ""),
            retrieval_context=context,
        )
        for metric in metrics:
            metric.measure(test_case)
            scores[metric.__class__.__name__].append(float(metric.score or 0.0))

    return scores


def _write_report(scores: dict[str, list[float]], n: int) -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    settings = get_settings()
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Generated: {now}",
        f"- Questions evaluated: {n}",
        f"- LLM: `{settings.llm_model}`  |  Embeddings: `{settings.embedding_model}`",
        f"- Retrieval: `{settings.retrieval_strategy}`, top_k=`{settings.top_k}`",
        "",
        "| Metric | Mean | Min | Max |",
        "|---|---|---|---|",
    ]
    for name, values in scores.items():
        if not values:
            continue
        lines.append(
            f"| {name} | {statistics.mean(values):.3f} "
            f"| {min(values):.3f} | {max(values):.3f} |"
        )
    lines.append("")
    lines.append(
        "_Scores are produced by an LLM judge (DeepEval) and are non-deterministic._"
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("report_written", path=str(REPORT_PATH))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with DeepEval.")
    parser.add_argument(
        "--smoke", action="store_true", help=f"Only evaluate {SMOKE_COUNT} questions."
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    enable_system_trust_store()

    items = _load_dataset(args.smoke)
    log.info("eval_start", count=len(items), smoke=args.smoke)
    scores = _evaluate(items)
    _write_report(scores, len(items))

    # Console summary.
    for name, values in scores.items():
        if values:
            print(f"{name}: mean={statistics.mean(values):.3f} (n={len(values)})")


if __name__ == "__main__":
    main()
