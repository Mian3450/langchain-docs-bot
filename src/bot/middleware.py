"""aiogram middlewares: per-user rate limiting and structured request logging.

Rate limiting uses an in-memory sliding window keyed by user ID — sufficient for
a single-process demo (no Redis). For multi-replica deployments this would need
a shared store.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

log = structlog.get_logger(__name__)

_WINDOW_SECONDS = 60.0


class RateLimitMiddleware(BaseMiddleware):
    """Reject a user's messages once they exceed N per rolling minute."""

    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._hits: dict[int, deque[float]] = defaultdict(deque)

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message) and event.from_user is not None:
            user_id = event.from_user.id
            now = time.monotonic()
            hits = self._hits[user_id]
            while hits and now - hits[0] > _WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= self._limit:
                log.warning("rate_limited", user_id=user_id, limit=self._limit)
                await event.answer(
                    f"⏳ Rate limit reached ({self._limit}/min). "
                    "Please wait a moment before asking again."
                )
                return None
            hits.append(now)
        return await handler(event, data)


class LoggingMiddleware(BaseMiddleware):
    """Log each message (user ID, text length) and the handler response time."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message):
            return await handler(event, data)

        user_id = event.from_user.id if event.from_user else None
        text = event.text or ""
        structlog.contextvars.bind_contextvars(user_id=user_id)
        start = time.perf_counter()
        try:
            log.info("message_received", text_len=len(text), preview=text[:80])
            return await handler(event, data)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            log.info("message_handled", elapsed_ms=round(elapsed_ms, 1))
            structlog.contextvars.clear_contextvars()
