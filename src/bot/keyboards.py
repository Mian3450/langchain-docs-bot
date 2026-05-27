"""Inline keyboards (optional UI sugar)."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def repo_keyboard(repo_url: str) -> InlineKeyboardMarkup:
    """A single-button keyboard linking to the project's source repository."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="⭐ Source on GitHub", url=repo_url)]]
    )
