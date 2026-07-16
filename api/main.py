"""FastAPI routes that depend only on abstract application ports."""

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from core.container import ApplicationContainer
from core.ports import AudioResource, AudioStoragePort
from models.schemas import AnalysisResponse

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def get_container(request: Request) -> ApplicationContainer:
    return request.app.state.container


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    container: ApplicationContainer = Depends(get_container),
) -> AnalysisResponse:
    audio = await _resolve_audio(request, container.audio_storage)
    try:
        analysis_response = await container.analysis.analyze(audio.path)
    finally:
        audio.close()
    return analysis_response


async def _resolve_audio(
    request: Request,
    storage: AudioStoragePort
) -> AudioResource:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        audio = await _multipart_audio(request, storage)
    elif content_type.startswith("application/json"):
        audio = await _json_audio(request, storage)
    else:
        raise ValueError("Используйте multipart/form-data или application/json")
    return audio


async def _multipart_audio(
    request: Request,
    storage: AudioStoragePort
) -> AudioResource:
    form = await request.form()
    file, url = form.get("file"), form.get("url")
    if isinstance(file, UploadFile):
        audio = await storage.from_upload(file)
    elif url:
        audio = await storage.from_url(str(url))
    else:
        raise ValueError("Передайте аудиофайл в `file` или ссылку в `url`")
    return audio


async def _json_audio(
    request: Request, storage: AudioStoragePort
) -> AudioResource:
    payload = await request.json()
    if url := payload.get("url"):
        audio = await storage.from_url(str(url))
    else:
        raise ValueError("Передайте ссылку в поле `url`")
    return audio


async def value_error_handler(_request: Request, error: ValueError):
    error_response = JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(error)},
    )
    return error_response
