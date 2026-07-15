"""Safe temporary storage for uploaded or remote audio."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

import httpx
from fastapi import UploadFile

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg"}


def validate_audio_name(name: str) -> str:
    suffix = Path(urlparse(name).path).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Неподдерживаемый формат аудио. Допустимы: {supported}")
    return suffix


async def save_upload(file: UploadFile, max_bytes: int) -> Path:
    suffix = validate_audio_name(file.filename or "")
    return await _write_chunks(file, suffix, max_bytes)


async def download_audio(url: str, max_bytes: int) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL аудио должен использовать http или https")
    suffix = validate_audio_name(url)

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with NamedTemporaryFile(delete=False, suffix=suffix) as target:
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        Path(target.name).unlink(missing_ok=True)
                        raise ValueError("Аудиофайл превышает допустимый размер")
                    target.write(chunk)
                return Path(target.name)


async def _write_chunks(file: UploadFile, suffix: str, max_bytes: int) -> Path:
    with NamedTemporaryFile(delete=False, suffix=suffix) as target:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                Path(target.name).unlink(missing_ok=True)
                raise ValueError("Аудиофайл превышает допустимый размер")
            target.write(chunk)
        return Path(target.name)
