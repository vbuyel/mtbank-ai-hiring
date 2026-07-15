"""Produces the final summary using peer-agent findings as context."""

import json

from agents.base import BaseAgent
from models.schemas import (
    ClassificationResult,
    ComplianceResult,
    QualityResult,
    SummaryResult,
    TranscriptSegment,
)


class SummarizerAgent(BaseAgent):
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
        *,
        classification: ClassificationResult,
        quality: QualityResult,
        compliance: ComplianceResult,
    ) -> SummaryResult:
        peer_results = {
            "classification": classification.model_dump(mode="json"),
            "quality": quality.model_dump(mode="json"),
            "compliance": compliance.model_dump(mode="json"),
        }
        context = (
            "Результаты других агентов:\n"
            f"{json.dumps(peer_results, ensure_ascii=False)}"
        )
        return await self._execute(transcript, SummaryResult, context)
