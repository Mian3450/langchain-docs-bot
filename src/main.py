"""Entry point: configure logging, wire up the bot, and start long polling.

Run with ``python -m src.main`` (this is the Docker container command too).
"""

from __future__ import annotations

import asyncio

import structlog
from aiogram import Bot, Dispatcher

from src.bot.handlers import router
from src.bot.middleware import LoggingMiddleware, RateLimitMiddleware
from src.config import get_settings
from src.utils.logging import configure_logging
from src.utils.tls import enable_system_trust_store

log = structlog.get_logger(__name__)


def build_dispatcher(rate_limit_per_minute: int) -> Dispatcher:
    """Create a Dispatcher with middlewares and handlers registered."""
    dp = Dispatcher()
    # Logging outermost so it wraps (and times) the rate-limit check too.
    dp.message.middleware(LoggingMiddleware())
    dp.message.middleware(RateLimitMiddleware(rate_limit_per_minute))
    dp.include_router(router)
    return dp


async def run() -> None:
    """Start the bot and poll until interrupted."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    enable_system_trust_store()

    bot = Bot(token=settings.telegram_bot_token.get_secret_value())
    dp = build_dispatcher(settings.rate_limit_per_minute)

    log.info("bot_starting", llm=settings.llm_model, strategy=settings.retrieval_strategy)
    try:
        # Drop any updates that piled up while the bot was offline.
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        log.info("bot_stopped")


def main() -> None:
    """Synchronous wrapper for the console / module entry point."""
    try:
        asyncio.run(run())
    except (KeyboardInterrupt, SystemExit):
        log.info("interrupted")


if __name__ == "__main__":
    main()
