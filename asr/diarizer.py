"""Baseline role assignment with a replaceable diarization strategy."""

from core.ports import DiarizerPort
from models.schemas import RawSegment, Speaker, TranscriptSegment

OPERATOR_MARKERS = (
    "мтбанк",
    "меня зовут",
    "чем могу помочь",
)


class Diarizer(DiarizerPort):
    """Assign roles by turn alternation.

    This baseline is deterministic and keeps the architecture usable without
    a second ML model. Replace it with pyannote in production.
    """


    def assign_speakers(
        self, segments: list[RawSegment]
    ) -> list[TranscriptSegment]:
        if not segments:
            return []
        first_speaker = self._detect_first_speaker(segments[0].text)
        other_speaker = "Клиент" if first_speaker == "Оператор" else "Оператор"
        speakers = (first_speaker, other_speaker)
        assigned_speakers = [
            self._with_speaker(segment, speakers[index % 2])
            for index, segment in enumerate(segments)
        ]
        return assigned_speakers


    @staticmethod
    def _with_speaker(segment: RawSegment, speaker: Speaker) -> TranscriptSegment:
        segment = TranscriptSegment(
            speaker=speaker,
            start=segment.start,
            end=segment.end,
            text=segment.text,
        )
        return segment


    @staticmethod
    def _detect_first_speaker(text: str) -> Speaker:
        normalized = text.lower()
        if any(marker in normalized for marker in OPERATOR_MARKERS):
            return "Оператор"
        return "Клиент"
