"""Classifies the call topic and operational priority."""

from agents.base import BaseAgent
from core.ports import ClassifierPort
from models.schemas import ClassificationResult, TranscriptSegment


class ClassifierAgent(BaseAgent, ClassifierPort):
    name = "classifier"

    def system_prompt(self) -> str:
        return (
            "Ты классификатор звонков банка. Определи основную тему обращения "
            "и приоритет. High используй для мошенничества, блокировки денег, "
            "эскалации или серьёзной жалобы."
        )

    async def run(
        self, transcript: list[TranscriptSegment]
    ) -> ClassificationResult:
        classification_result = await self._execute(transcript, ClassificationResult)
        return classification_result
