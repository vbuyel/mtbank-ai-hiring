"""Classifies the call topic and operational priority."""

from agents.base import BaseAgent
from models.schemas import ClassificationResult, TranscriptSegment


class ClassifierAgent(BaseAgent):
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
        return await self._execute(transcript, ClassificationResult)
