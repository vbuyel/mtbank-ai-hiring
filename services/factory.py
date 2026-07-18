"""Composition root: the only place that wires concrete dependencies."""

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent
from asr.diarizer import Diarizer
from asr.transcriber import Transcriber
from settings import Settings
from core.container import ApplicationContainer
from services.analysis import AnalysisDependencies, AnalysisService
from services.llm_client import LLMClient
from shared.audio import LocalAudioStorage


def build_application(settings: Settings) -> ApplicationContainer:
    dependencies = _build_analysis_dependencies(settings)
    app_container = ApplicationContainer(
        analysis=AnalysisService(dependencies),
        audio_storage=LocalAudioStorage(settings.max_audio_bytes),
    )
    return app_container


def _task_llm(settings: Settings, prefix: str) -> LLMClient:
    return LLMClient(
        base_url=getattr(settings, f"{prefix}_llm_base_url"),
        api_key=getattr(settings, f"{prefix}_llm_api_key"),
        model=getattr(settings, f"{prefix}_llm_model"),
    )


def _build_analysis_dependencies(settings: Settings) -> AnalysisDependencies:
    transcriber = Transcriber(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    return AnalysisDependencies(
        transcriber=transcriber,
        diarizer=Diarizer(_task_llm(settings, "diarizer")),
        classifier=ClassifierAgent(_task_llm(settings, "classifier")),
        quality=QualityAgent(_task_llm(settings, "quality")),
        compliance=ComplianceAgent(_task_llm(settings, "compliance")),
        summarizer=SummarizerAgent(_task_llm(settings, "summarizer")),
    )
