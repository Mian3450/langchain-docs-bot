"""structlog configuration.

Call :func:`configure_logging` once at process start (e.g. in ``main.py`` or a
script entry point). Everywhere else, obtain a logger with
``structlog.get_logger(__name__)``.

The setup routes both structlog *and* stdlib logging (aiogram, httpx, chromadb,
...) through a single :class:`structlog.stdlib.ProcessorFormatter`, so all log
lines share one rendering style.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure structlog + stdlib logging for the whole process.

    Args:
        level: Standard logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        fmt: ``"console"`` for human-readable colourised output (local dev) or
            ``"json"`` for structured single-line JSON (production / Docker).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer: structlog.typing.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # structlog-native loggers: build the event dict, then hand off to the
    # stdlib ProcessorFormatter for final rendering.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Single stdlib handler that renders both structlog and foreign records.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level)

    # Quiet noisy third-party loggers; let our level govern the rest.
    for noisy in ("httpx", "httpcore", "chromadb", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))
