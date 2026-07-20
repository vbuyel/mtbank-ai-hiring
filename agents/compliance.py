"""Detects potentially non-compliant operator statements."""

from agents.base import BaseAgent
from agents.validation import is_operator_quote
from core.ports import CompliancePort
from models.schemas import ComplianceResult, TranscriptSegment


class ComplianceAgent(BaseAgent, CompliancePort):
    name = "compliance"

    def system_prompt(self) -> str:
        return """Проверь только реплики Оператора: гарантии одобрения,
гарантированная доходность, запрос PIN/CVV/пароля/SMS-кода, раскрытие чужих
данных и утверждения, явно опровергнутые транскриптом. Пропуск информации сам
по себе не нарушение. Для каждого нарушения приведи точную непустую цитату
Оператора; без такой цитаты не добавляй issue. Если issues пуст, passed=true."""

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
