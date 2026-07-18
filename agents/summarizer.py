"""Produces a call summary from the transcript."""

from agents.base import BaseAgent
from core.ports import SummarizerPort
from models.schemas import (
    SummaryResult,
    TranscriptSegment,
)


class SummarizerAgent(BaseAgent, SummarizerPort):
    name = "summarizer"

    def system_prompt(self) -> str:
        return (
            "Ты суммаризатор звонка контакт-центра. Напиши нейтральное резюме "
            "в 3–5 предложениях и конкретные action items. Используй только "
            "факты из транскрипта."
        )

    async def run(
        self,
        transcript: list[TranscriptSegment],
    ) -> SummaryResult:
        return await self._execute(transcript, SummaryResult)
