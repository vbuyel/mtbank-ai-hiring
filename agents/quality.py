"""Scores the operator against a compact service checklist."""

from agents.base import BaseAgent
from core.ports import QualityPort
from models.schemas import QualityResult, TranscriptSegment


class QualityAgent(BaseAgent, QualityPort):
    name = "quality"
    weights = {
        "greeting": 25,
        "need_detection": 25,
        "solution_provided": 35,
        "farewell": 15,
    }

    def system_prompt(self) -> str:
        return (
            "Ты контролёр качества контакт-центра. Проверь только реплики "
            "оператора: приветствие, выявление потребности, предложенное решение "
            "и прощание. Решение считается предоставленным только если оператор "
            "сообщил конкретный ответ или следующий шаг. Поле total будет "
            "пересчитано программно."
        )

    async def run(self, transcript: list[TranscriptSegment]) -> QualityResult:
        result = await self._execute(transcript, QualityResult)
        total = sum(
            self.weights[name]
            for name, value in result.checklist.model_dump().items()
            if value
        )
        return result.model_copy(update={"total": total})
