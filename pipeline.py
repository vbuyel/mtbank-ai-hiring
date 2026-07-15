"""
title: MTBank Call Analytics
author: Vladislav Buyel
version: 0.1.0
license: MIT
description: ASR and four-agent analysis for Russian contact-center calls.
"""

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config import Settings, get_settings
from models.schemas import AnalysisResponse
from services.analysis import AnalysisService
from services.factory import build_analysis_service
from utils.audio import download_audio
from utils.logging import configure_logging

AUDIO_URL_RE = re.compile(r"https?://\S+\.(?:wav|mp3|ogg)(?:\?\S*)?", re.IGNORECASE)


class Pipeline:
    class Valves(BaseModel):
        LLM_BASE_URL: str = "http://host.docker.internal:11434/v1"
        LLM_API_KEY: str = Field(default="ollama", json_schema_extra={"secret": True})
        LLM_MODEL: str = "qwen2.5:7b"
        WHISPER_MODEL: str = "medium"
        WHISPER_DEVICE: str = "cpu"
        WHISPER_COMPUTE_TYPE: str = "int8"
        MAX_AUDIO_BYTES: int = 50 * 1024 * 1024

    def __init__(self) -> None:
        settings = get_settings()
        self.name = "MTBank Call Analytics"
        self.valves = self.Valves(
            LLM_BASE_URL=settings.llm_base_url,
            LLM_API_KEY=settings.llm_api_key,
            LLM_MODEL=settings.llm_model,
            WHISPER_MODEL=settings.whisper_model,
            WHISPER_DEVICE=settings.whisper_device,
            WHISPER_COMPUTE_TYPE=settings.whisper_compute_type,
            MAX_AUDIO_BYTES=settings.max_audio_bytes,
        )
        self.service: AnalysisService | None = None

    async def on_startup(self) -> None:
        configure_logging()
        settings = Settings(
            llm_base_url=self.valves.LLM_BASE_URL,
            llm_api_key=self.valves.LLM_API_KEY,
            llm_model=self.valves.LLM_MODEL,
            whisper_model=self.valves.WHISPER_MODEL,
            whisper_device=self.valves.WHISPER_DEVICE,
            whisper_compute_type=self.valves.WHISPER_COMPUTE_TYPE,
            max_audio_bytes=self.valves.MAX_AUDIO_BYTES,
        )
        self.service = build_analysis_service(settings)

    async def on_shutdown(self) -> None:
        self.service = None

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
    ) -> str:
        del __user__
        if self.service is None:
            await self.on_startup()

        audio_reference = self._extract_audio_reference(body)
        temporary_path: Path | None = None
        if audio_reference.startswith(("http://", "https://")):
            temporary_path = await download_audio(
                audio_reference,
                self.valves.MAX_AUDIO_BYTES,
            )
            audio_path = temporary_path
        else:
            audio_path = Path(audio_reference)

        try:
            assert self.service is not None
            result = await self.service.analyze(audio_path)
            return self._format_markdown(result)
        finally:
            if temporary_path:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _extract_audio_reference(body: dict[str, Any]) -> str:
        if audio_url := body.get("audio_url"):
            return str(audio_url)

        for file_info in body.get("files", []):
            if reference := file_info.get("path") or file_info.get("url"):
                return str(reference)

        messages = body.get("messages", [])
        content = str(messages[-1].get("content", "")) if messages else ""
        if match := AUDIO_URL_RE.search(content):
            return match.group(0)
        raise ValueError("Загрузите WAV/MP3/OGG файл или отправьте прямой URL")

    @staticmethod
    def _format_markdown(result: AnalysisResponse) -> str:
        transcript = "\n".join(
            f"- `{item.start:.1f}–{item.end:.1f}` **{item.speaker}:** {item.text}"
            for item in result.transcript
        )
        issues = (
            "\n".join(f"- {issue.rule}: “{issue.quote}”" for issue in result.compliance.issues)
            or "- Нарушений не найдено"
        )
        actions = "\n".join(f"- {item}" for item in result.action_items) or "- Нет"
        return (
            "## Анализ звонка\n"
            f"**Тема:** {result.classification.topic}  \n"
            f"**Приоритет:** {result.classification.priority}  \n"
            f"**Качество:** {result.quality_score.total}/100\n\n"
            f"### Резюме\n{result.summary}\n\n"
            f"### Compliance\n{issues}\n\n"
            f"### Дальнейшие действия\n{actions}\n\n"
            f"### Транскрипт\n{transcript}"
        )
