"""Produces the final summary using peer-agent findings as context."""

import json

from agents.base import BaseAgent
from core.ports import SummarizerPort
from models.schemas import (
    SummaryContext,
    SummaryResult,
    TranscriptSegment,
)


class SummarizerAgent(BaseAgent, SummarizerPort):
    name = "summarizer"

    def system_prompt(self) -> str:
        return (
            "Ты суммаризатор звонка контакт-центра. Напиши нейтральное резюме "
            "в 3–5 предложениях и конкретные action items. Учитывай результаты "
            "других агентов, но не добавляй фактов, которых нет в транскрипте."
        )

    async def run(
        self,
        transcript: list[TranscriptSegment],
        context: SummaryContext,
    ) -> SummaryResult:
        peer_results = context.model_dump(mode="json")
        prompt_context = (
            "Результаты других агентов:\n"
            f"{json.dumps(peer_results, ensure_ascii=False)}"
        )
        return await self._execute(transcript, SummaryResult, prompt_context)
