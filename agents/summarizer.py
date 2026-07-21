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
        return """Ты суммаризатор контакт-центра. Составь нейтральное резюме звонка в 3–5 предложениях:
причина обращения, существенные факты, ответ оператора и итог. Используй только
транскрипт. В action_items включай лишь явно согласованные следующие действия;
если их нет, верни пустой список."""

    async def run(
        self,
        transcript: list[TranscriptSegment],
    ) -> SummaryResult:
        return await self._execute(transcript, SummaryResult)
