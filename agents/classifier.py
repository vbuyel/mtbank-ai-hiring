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
        result = await self._execute(transcript, ClassificationResult)
        text = " ".join(item.text for item in transcript).casefold()
        complaint_markers = ("жалоб", "недовол", "претенз", "возмущ", "обман")
        false_complaint = not any(marker in text for marker in complaint_markers)
        if result.topic == "жалобы" and "кредит" in text and false_complaint:
            return result.model_copy(update={"topic": "кредиты"})
        return result
