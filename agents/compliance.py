"""Detects potentially non-compliant operator statements."""

from agents.base import BaseAgent
from agents.validation import is_operator_quote
from core.ports import CompliancePort
from models.schemas import ComplianceResult, TranscriptSegment


class ComplianceAgent(BaseAgent, CompliancePort):
    name = "compliance"

    def system_prompt(self) -> str:
        return (
            "Ты специалист банковского compliance. Проверяй реплики оператора "
            "на гарантии одобрения, обещание гарантированной доходности, раскрытие "
            "чувствительных данных и явно недостоверные утверждения. Отсутствие "
            "информации само по себе не является нарушением. Для каждого нарушения "
            "приводи непустую точную цитату оператора. Цитата должна дословно "
            "присутствовать в транскрипте. Не выдумывай нарушения."
        )

    async def run(
        self, transcript: list[TranscriptSegment]
    ) -> ComplianceResult:
        result = await self._execute(transcript, ComplianceResult)
        issues = [
            issue
            for issue in result.issues
            if is_operator_quote(issue.quote, transcript)
        ]
        return result.model_copy(update={"passed": not issues, "issues": issues})
