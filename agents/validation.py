"""Deterministic validation helpers for evidence returned by agents."""

import re

from models.schemas import TranscriptSegment


def normalize_quote(text: str) -> str:
    stripped = text.strip(" \t\n\"'«»“”„")
    return re.sub(r"\s+", " ", stripped).casefold()


def is_operator_quote(quote: str, transcript: list[TranscriptSegment]) -> bool:
    expected = normalize_quote(quote)
    operator_lines = (
        normalize_quote(item.text)
        for item in transcript
        if item.speaker == "Оператор"
    )
    return bool(expected) and any(expected in line for line in operator_lines)
