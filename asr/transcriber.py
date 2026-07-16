"""Adapter around faster-whisper with async-friendly execution."""

import asyncio
from pathlib import Path
from faster_whisper import WhisperModel

from core.ports import TranscriberPort
from models.schemas import RawSegment
from utils.audio import validate_audio_name


class Transcriber(TranscriberPort):
    def __init__(
        self,
        model_name: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
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
        transctiption = await asyncio.to_thread(self._transcribe_sync, path)
        return transctiption


    def _transcribe_sync(self, path: Path) -> list[RawSegment]:
        segments, _ = self._model.transcribe(
            str(path),
            language="ru",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        raw_segmants = self._to_raw_segments(segments)
        return raw_segmants


    @staticmethod
    def _to_raw_segments(segments) -> list[RawSegment]:
        raw_segments = [
            RawSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in segments
            if segment.text.strip()
        ]
        return raw_segments
