"""Detects potentially non-compliant operator statements."""

from agents.base import BaseAgent
from models.schemas import ComplianceResult, TranscriptSegment


class ComplianceAgent(BaseAgent):
    name = "compliance"

    def system_prompt(self) -> str:
        return (
            "Ты специалист банковского compliance. Проверяй реплики оператора "
            "на гарантии одобрения, обещание гарантированной доходности, раскрытие "
            "чувствительных данных и отсутствие обязательных предупреждений. "
            "Для каждого нарушения приводи точную цитату. Не выдумывай нарушения."
        )

    async def run(
        self, transcript: list[TranscriptSegment]
    ) -> ComplianceResult:
        return await self._execute(transcript, ComplianceResult)
