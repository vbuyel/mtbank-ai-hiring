"""Baseline role assignment with a replaceable diarization strategy."""

from models.schemas import RawSegment, TranscriptSegment

OPERATOR_MARKERS = (
    "добрый день",
    "здравствуйте",
    "мтбанк",
    "меня зовут",
    "чем могу помочь",
)


class Diarizer:
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

        result: list[TranscriptSegment] = []
        for index, segment in enumerate(segments):
            speaker = first_speaker if index % 2 == 0 else other_speaker
            result.append(
                TranscriptSegment(
                    speaker=speaker,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                )
            )
        return result

    @staticmethod
    def _detect_first_speaker(text: str) -> str:
        normalized = text.lower()
        if any(marker in normalized for marker in OPERATOR_MARKERS):
            return "Оператор"
        return "Клиент"
