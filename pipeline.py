"""
title: MTBank Call Analytics
author: Vladislav Buyel
version: 0.1.0
license: MIT
description: ASR and four-agent analysis for Russian contact-center calls.
"""

import asyncio
import re
from threading import Lock
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from settings import Settings, get_settings
from core.container import ApplicationContainer
from core.ports import AudioResource
from models.schemas import AnalysisResponse
from services.factory import build_application
from shared.logging import configure_logging

AUDIO_URL_RE = re.compile(r"https?://\S+\.(?:wav|mp3|ogg)(?:\?\S*)?", re.IGNORECASE)
WEBUI_UPLOAD_DIR = Path("/open-webui-data/uploads")


@dataclass
class LocalAudio(AudioResource):
    _path: Path

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        pass


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
        self.container: ApplicationContainer | None = None
        self._runner = asyncio.Runner()
        self._runner_lock = Lock()


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
        self.container = build_application(settings)


    async def on_shutdown(self) -> None:
        self.container = None


    async def inlet(
        self, body: dict[str, Any], user: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        del user
        task = body.get("metadata", {}).get("task")
        if task:
            body["task"] = task
            body["skip_audio"] = True
            return body
        audio_path = self._upload_path(body)
        if audio_path is not None:
            body["audio_url"] = str(audio_path)
        else:
            body["skip_audio"] = True
        return body


    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
    ) -> str:
        del model_id, messages
        if (
            body.get("skip_audio")
            or body.get("task")
            or body.get("metadata", {}).get("task")
        ):
            return user_message
        with self._runner_lock:
            return self._runner.run(self._analyze(body))


    async def _analyze(self, body: dict[str, Any]) -> str:
        if self.container is None:
            await self.on_startup()
        assert self.container is not None
        audio = await self._resolve_audio(body)
        try:
            return self._format_markdown(await self.container.analysis.analyze(audio.path))
        finally:
            audio.close()


    async def _resolve_audio(self, body: dict[str, Any]) -> AudioResource:
        reference = self._extract_audio_reference(body)
        assert self.container is not None
        if reference.startswith(("http://", "https://")):
            return await self.container.audio_storage.from_url(reference)
        return LocalAudio(Path(reference))


    @staticmethod
    def _extract_audio_reference(body: dict[str, Any]) -> str:
        if audio_url := body.get("audio_url"):
            return str(audio_url)
        messages = body.get("messages", [])
        content = str(messages[-1].get("content", "")) if messages else ""
        if match := AUDIO_URL_RE.search(content):
            return match.group(0)
        raise ValueError("Загрузите WAV/MP3/OGG файл или отправьте прямой URL")


    @staticmethod
    def _upload_path(body: dict[str, Any]) -> Path | None:
        metadata = body.get("metadata", {})
        for attachment in metadata.get("files", []):
            file = attachment["file"]
            path = WEBUI_UPLOAD_DIR / f"{file['id']}_{Path(file['filename']).name}"
            if path.is_file():
                return path
        return None


    @staticmethod
    def _format_markdown(result: AnalysisResponse) -> str:
        transcript = Pipeline._format_transcript(result)
        issues = Pipeline._format_issues(result)
        actions = Pipeline._format_actions(result)
        formatted_result = (
            "## Анализ звонка\n"
            f"**Тема:** {result.classification.topic}  \n"
            f"**Приоритет:** {result.classification.priority}  \n"
            f"**Качество:** {result.quality_score.total}/100\n\n"
            f"### Резюме\n{result.summary}\n\n"
            f"### Compliance\n{issues}\n\n"
            f"### Дальнейшие действия\n{actions}\n\n"
            f"### Транскрипт\n{transcript}"
        )
        return formatted_result


    @staticmethod
    def _format_transcript(result: AnalysisResponse) -> str:
        formatted_transcrition = "\n".join(
            f"- `{item.start:.1f}–{item.end:.1f}` **{item.speaker}:** {item.text}"
            for item in result.transcript
        )
        return formatted_transcrition


    @staticmethod
    def _format_issues(result: AnalysisResponse) -> str:
        formatted_issues = "\n".join(
            f"- {issue.rule}: “{issue.quote}”"
            for issue in result.compliance.issues
        ) or "- Нарушений не найдено"
        return formatted_issues


    @staticmethod
    def _format_actions(result: AnalysisResponse) -> str:
        formatted_actions = "\n".join(f"- {item}" for item in result.action_items) or "- Нет"
        return formatted_actions
