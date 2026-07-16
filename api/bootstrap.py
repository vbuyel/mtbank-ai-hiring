"""HTTP composition root; concrete adapters are wired only here."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.main import router, value_error_handler
from config import get_settings
from services.factory import build_application
from utils.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.container = build_application(settings)
    yield
    app.state.container = None


def create_app() -> FastAPI:
    application = FastAPI(
        title="MTBank Call Analytics API",
        lifespan=lifespan,
    )
    application.include_router(router)
    application.add_exception_handler(ValueError, value_error_handler)
    return application


app = create_app()
