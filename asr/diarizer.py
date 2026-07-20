"""LLM-based role assignment to partition dialogues between Operator and Client."""

from core.ports import DiarizerPort, StructuredLLMPort
from models.schemas import DiarizationResult, RawSegment, Speaker, TranscriptSegment


class Diarizer(DiarizerPort):
    """Assign roles by reconciling independent LLM and heuristic markup."""

    operator_hints = (
        "чем могу помочь", "могу предложить",
        "вас интересует", "вам удобно", "составит около",
        "подскажите ваш", "есть еще вопросы",
        "мтбанк", "меня зовут",
        "пожалуйста",
    )
    client_hints = (
        "хочу узнать", "мне нуж", "лучше онлайн",
        "связаться с отдел", "кто у вас",
        "как я могу", "уточнить",
        "спасибо за инфор",
    )

    def __init__(self, llm: StructuredLLMPort) -> None:
        self.llm = llm


    async def assign_speakers(
        self, segments: list[RawSegment]
    ) -> list[TranscriptSegment]:
        if not segments:
            return []
        heuristic = self._heuristic_speakers(segments)
        llm_speakers = await self._call_llm_for_speakers(segments)
        speakers = await self._reconcile_speakers(
            segments, llm_speakers, heuristic
        )
        return self._transcript(segments, speakers)


    @staticmethod
    def _transcript(
        segments: list[RawSegment], speakers: list[Speaker]
    ) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                speaker=speakers[i],
                start=seg.start, end=seg.end, text=seg.text
            )
            for i, seg in enumerate(segments)
        ]


    def _heuristic_speaker(self, text: str, index: int) -> Speaker:
        normalized = text.casefold()
        if any(hint in normalized for hint in self.operator_hints):
            return "Оператор"
        if any(hint in normalized for hint in self.client_hints):
            return "Клиент"
        return "Оператор" if index % 2 == 0 else "Клиент"


    def _heuristic_speakers(
        self, segments: list[RawSegment]
    ) -> list[Speaker]:
        return [
            self._heuristic_speaker(segment.text, index)
            for index, segment in enumerate(segments)
        ]


    async def _call_llm_for_speakers(self, segments: list[RawSegment]) -> list[Speaker]:
        formatted = "\n".join(f"{i}: {seg.text}" for i, seg in enumerate(segments))
        result = await self.llm.complete_json(
            system_prompt=self._initial_system_prompt(),
            user_prompt=f"Разметь по ролям следующие реплики:\n{formatted}",
            response_model=DiarizationResult,
        )
        fallback: list[Speaker] = ["Клиент"] * len(segments)
        return self._speakers_from(result, fallback)


    @staticmethod
    def _speakers_from(
        result: DiarizationResult, fallback: list[Speaker]
    ) -> list[Speaker]:
        mapping = {item.index: item.speaker for item in result.speakers}
        return [mapping.get(i, speaker) for i, speaker in enumerate(fallback)]


    async def _reconcile_speakers(
        self, segments: list[RawSegment],
        llm_speakers: list[Speaker],
        heuristic: list[Speaker],
    ) -> list[Speaker]:
        result = await self.llm.complete_json(
            system_prompt=self._reconciliation_system_prompt(),
            user_prompt=self._reconciliation_prompt(
                segments, llm_speakers, heuristic
            ),
            response_model=DiarizationResult,
        )
        return self._speakers_from(result, llm_speakers)


    @staticmethod
    def _reconciliation_prompt(
        segments: list[RawSegment],
        llm_speakers: list[Speaker],
        heuristic: list[Speaker],
    ) -> str:
        dialogue = "\n".join(f"{i}: {seg.text}" for i, seg in enumerate(segments))
        llm_markup = "\n".join(f"{i}: {role}" for i, role in enumerate(llm_speakers))
        heuristic_markup = "\n".join(f"{i}: {role}" for i, role in enumerate(heuristic))
        return (
            f"Реплики:\n{dialogue}\n\nНезависимая разметка LLM:\n{llm_markup}\n\n"
            f"Эвристическая разметка:\n{heuristic_markup}\n\n"
            "Сверь обе разметки и верни окончательную."
        )


    @staticmethod
    def _initial_system_prompt() -> str:
        return """Разметь ASR-сегменты входящего звонка в банк: Оператор или Клиент.
Оператор обычно говорит первым, здоровается, называет банк и своё имя — это
сильный приоритет, но не абсолютное правило. Главнее смысл: оператор выясняет
запрос, объясняет условия и даёт инструкции; клиент описывает ситуацию и
отвечает о себе. Учитывай связность: обрывки одной фразы имеют одну роль, а
короткий ответ обычно принадлежит отвечающему. Имя или приветствие сами по себе
роль не определяют. Верни каждый index ровно один раз в исходном порядке."""


    @staticmethod
    def _reconciliation_system_prompt() -> str:
        return """Ты выполняешь финальную диаризацию банковского звонка.
Сравни свою независимую разметку с эвристической разметкой по ключевым фразам
и позиции реплики. Обе могут ошибаться, поэтому разрешай расхождения по смыслу
всего диалога: оператор представляет банк, выясняет запрос, объясняет условия и
даёт инструкции; клиент формулирует запрос и сообщает сведения о себе.
Сохраняй связность обрывков фраз. Верни окончательную роль каждого index ровно
один раз в исходном порядке: Оператор или Клиент."""
