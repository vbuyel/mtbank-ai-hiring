"""Safe temporary storage for uploaded or remote audio."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx

from core.ports import AudioResource, AudioStoragePort

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg"}


def validate_audio_name(name: str) -> str:
    suffix = Path(urlparse(name).path).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Неподдерживаемый формат аудио. Допустимы: {supported}")
    return suffix


@dataclass
class TemporaryAudio(AudioResource):
    _path: Path

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        self._path.unlink(missing_ok=True)


class LocalAudioStorage(AudioStoragePort):
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes


    async def from_upload(self, file: object) -> AudioResource:
        upload = file
        suffix = validate_audio_name(getattr(upload, "filename", "") or "")
        chunks = self._upload_chunks(upload)
        audio = TemporaryAudio(await self._write(chunks, suffix))
        return audio


    async def from_url(self, url: str) -> AudioResource:
        self._validate_url(url)
        suffix = validate_audio_name(url)
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                path = await self._write(response.aiter_bytes(), suffix)
        audio = TemporaryAudio(path)
        return audio


    async def _write(self, chunks: AsyncIterator[bytes], suffix: str) -> Path:
        with NamedTemporaryFile(delete=False, suffix=suffix) as target:
            size = 0
            async for chunk in chunks:
                size += len(chunk)
                self._check_size(size, Path(target.name))
                target.write(chunk)
            path = Path(target.name)
            return path


    async def _upload_chunks(self, file: Any) -> AsyncIterator[bytes]:
        while chunk := await file.read(1024 * 1024):
            yield chunk


    def _check_size(self, size: int, path: Path) -> None:
        if size <= self.max_bytes:
            return
        path.unlink(missing_ok=True)
        raise ValueError("Аудиофайл превышает допустимый размер")


    @staticmethod
    def _validate_url(url: str) -> None:
        if urlparse(url).scheme not in {"http", "https"}:
            raise ValueError("URL аудио должен использовать http или https")
