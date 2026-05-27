"""aiogram message handlers.

The default handler is intentionally thin: it forwards the message text to
:func:`src.rag.pipeline.answer_question` and renders the result. All RAG logic
lives behind that single call.
"""

from __future__ import annotations

import structlog
from aiogram import Router
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from src.bot.keyboards import repo_keyboard
from src.config import get_settings
from src.rag.pipeline import answer_question
from src.utils.source_formatter import format_reply, split_for_telegram

log = structlog.get_logger(__name__)

router = Router(name="docs-bot")

# Project repository, shown in /about.
PROJECT_REPO_URL = "https://github.com/Mian3450/langchain-docs-bot"

_WELCOME = (
    "👋 <b>LangChain Docs Assistant</b>\n\n"
    "Ask me anything about the LangChain framework and I'll answer from the "
    "official documentation, with links to the sources.\n\n"
    "Just send your question as a message. Try /help for examples."
)

_HELP = (
    "<b>How to use</b>\n\n"
    "Send a question about LangChain in plain English. I retrieve the most "
    "relevant documentation passages and answer using only those, citing each "
    "source as <code>[1]</code>, <code>[2]</code>, ...\n\n"
    "<b>Examples</b>\n"
    "• What is LCEL and why use it?\n"
    "• How do I build a retrieval chain?\n"
    "• What's the difference between an agent and a chain?\n"
    "• How do callbacks work?\n\n"
    "Commands: /start /help /about"
)

_ABOUT = (
    "<b>About</b>\n\n"
    "A portfolio demo of a production-grade RAG pipeline: LangChain + ChromaDB "
    "for retrieval, FastEmbed for local embeddings, Groq for generation, "
    "aiogram for the bot.\n\n"
    f'Source code: <a href="{PROJECT_REPO_URL}">{PROJECT_REPO_URL}</a>'
)


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Reply to /start with a welcome + usage hint."""
    await message.answer(_WELCOME, parse_mode=ParseMode.HTML)


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    """Reply to /help with usage and example questions."""
    await message.answer(_HELP, parse_mode=ParseMode.HTML)


@router.message(Command("about"))
async def handle_about(message: Message) -> None:
    """Reply to /about with project info and the repo link."""
    await message.answer(
        _ABOUT,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=repo_keyboard(PROJECT_REPO_URL),
    )


@router.message()
async def handle_question(message: Message) -> None:
    """Default handler: treat any text as a question and answer it."""
    question = (message.text or "").strip()
    if not question:
        await message.answer(
            "Please send a text question about LangChain.", parse_mode=ParseMode.HTML
        )
        return

    if message.bot is not None:
        await message.bot.send_chat_action(message.chat.id, "typing")
    result = await answer_question(question)
    reply = format_reply(result.text, result.sources)

    max_len = get_settings().max_message_length
    for part in split_for_telegram(reply, max_len):
        await message.answer(
            part, parse_mode=ParseMode.HTML, disable_web_page_preview=True
        )
