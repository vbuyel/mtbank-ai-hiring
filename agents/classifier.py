"""Classifies the call topic and operational priority."""

from agents.base import BaseAgent
from core.ports import ClassifierPort
from models.schemas import ClassificationResult, TranscriptSegment


class ClassifierAgent(BaseAgent, ClassifierPort):
    name = "classifier"

    def system_prompt(self) -> str:
        return """Определи основную цель звонка и её приоритет.
Тема: кредиты, карты, переводы, жалобы или другое. Выбирай жалобы только при
явном недовольстве или претензии. Приоритет high — мошенничество, потеря карты,
блокировка/утрата денег или требование срочной эскалации; medium — проблема,
требующая действий банка; иначе low. Верни только вывод по транскрипту."""

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
