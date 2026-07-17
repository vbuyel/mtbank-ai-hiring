"""LLM-based role assignment to partition dialogues between Operator and Client."""

from core.ports import DiarizerPort, StructuredLLMPort
from models.schemas import DiarizationResult, RawSegment, Speaker, TranscriptSegment


class Diarizer(DiarizerPort):
    """Assign roles to speakers contextually using a LLM client."""

    def __init__(self, llm: StructuredLLMPort) -> None:
        self.llm = llm


    async def assign_speakers(
        self, segments: list[RawSegment]
    ) -> list[TranscriptSegment]:
        if not segments:
            return []
        speakers = await self._call_llm_for_speakers(segments)
        return [
            TranscriptSegment(
                speaker=speakers[i] if i < len(speakers) else "Клиент",
                start=seg.start, end=seg.end, text=seg.text
            )
            for i, seg in enumerate(segments)
        ]


    async def _call_llm_for_speakers(self, segments: list[RawSegment]) -> list[Speaker]:
        formatted = "\n".join(f"{i}: {seg.text}" for i, seg in enumerate(segments))
        system = (
            "Ты размечаешь реплики банковского звонка МТБанк. "
            "Для каждой реплики с индексом верни ровно одну роль: "
            "'Оператор' или 'Клиент'.\n"
            "Оператор: сотрудник банка — представляется от имени МТБанка, уточняет потребность, объясняет продукты/условия, предлагает действия.\n"
            "Клиент: обратившийся — описывает проблему или запрос, задаёт вопросы, сообщает персональные данные, отвечает на уточнения.\n"
            "Смотри на смысл реплики в контексте всего диалога, не только на отдельные слова.\n"
        )
        res = await self.llm.complete_json(
            system_prompt=system,
            user_prompt=f"Разметь по ролям следующие реплики:\n{formatted}",
            response_model=DiarizationResult
        )
        mapping = {item.index: item.speaker for item in res.speakers}
        return [mapping.get(i, "Клиент") for i in range(len(segments))]
