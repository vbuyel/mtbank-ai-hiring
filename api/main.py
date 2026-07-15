"""FastAPI adapter exposing the shared analysis use case."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from starlette.datastructures import UploadFile

from config import get_settings
from models.schemas import AnalysisResponse
from services.analysis import AnalysisService
from services.factory import build_analysis_service
from utils.audio import download_audio, save_upload
from utils.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.analysis_service = build_analysis_service(settings)
    yield
    app.state.analysis_service = None


app = FastAPI(
    title="MTBank Call Analytics API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: Request) -> AnalysisResponse:
    """Accept multipart `file`/`url` or JSON `{"url": "https://..."}`."""
    settings = get_settings()
    temporary_path: Path | None = None

    try:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            file = form.get("file")
            url = form.get("url")
            if isinstance(file, UploadFile):
                temporary_path = await save_upload(file, settings.max_audio_bytes)
            elif url:
                temporary_path = await download_audio(
                    str(url),
                    settings.max_audio_bytes,
                )
        elif content_type.startswith("application/json"):
            payload = await request.json()
            if url := payload.get("url"):
                temporary_path = await download_audio(
                    str(url),
                    settings.max_audio_bytes,
                )
        else:
            raise ValueError("Используйте multipart/form-data или application/json")

        if temporary_path is None:
            raise ValueError("Передайте аудиофайл в `file` или ссылку в `url`")

        service: AnalysisService = request.app.state.analysis_service
        return await service.analyze(temporary_path)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось проанализировать аудио",
        ) from error
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
