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
            "Ты — ассистент по разметке звонков. Определи роль говорящего для каждой реплики.\n"
            "Роли: 'Оператор' (представляет МТБанк, здоровается первым, предлагает помощь) "
            "или 'Клиент' (задает вопросы, жалуется, просит информацию)."
        )
        res = await self.llm.complete_json(
            system_prompt=system,
            user_prompt=f"Разметь по ролям следующие реплики:\n{formatted}",
            response_model=DiarizationResult
        )
        mapping = {item.index: item.speaker for item in res.speakers}
        return [mapping.get(i, "Клиент") for i in range(len(segments))]
