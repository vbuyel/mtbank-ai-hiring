"""Scores the operator against a compact service checklist."""

from agents.base import BaseAgent
from models.schemas import QualityResult, TranscriptSegment


class QualityAgent(BaseAgent):
    name = "quality"

    def system_prompt(self) -> str:
        return (
            "Ты контролёр качества контакт-центра. Проверь только реплики "
            "оператора: приветствие, выявление потребности, предложенное решение "
            "и прощание. Рассчитай аргументированный балл от 0 до 100."
        )

    async def run(self, transcript: list[TranscriptSegment]) -> QualityResult:
        return await self._execute(transcript, QualityResult)
