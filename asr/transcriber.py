"""Adapter around faster-whisper with async-friendly execution."""

import asyncio
from pathlib import Path

from models.schemas import RawSegment
from utils.audio import validate_audio_name


class Transcriber:
    def __init__(
        self,
        model_name: str = "medium",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    async def transcribe(self, audio_path: str | Path) -> list[RawSegment]:
        path = Path(audio_path)
        validate_audio_name(path.name)
        if not path.is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {path}")
        return await asyncio.to_thread(self._transcribe_sync, path)

    def _transcribe_sync(self, path: Path) -> list[RawSegment]:
        segments, _ = self._model.transcribe(
            str(path),
            language="ru",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return [
            RawSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in segments
            if segment.text.strip()
        ]
