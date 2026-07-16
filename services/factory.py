"""Composition root: the only place that wires concrete dependencies."""

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent
from asr.diarizer import Diarizer
from asr.transcriber import Transcriber
from settings import Settings
from core.container import ApplicationContainer
from core.ports import StructuredLLMPort
from services.analysis import AnalysisDependencies, AnalysisService
from services.llm_client import LLMClient
from shared.audio import LocalAudioStorage


def build_application(settings: Settings) -> ApplicationContainer:
    llm = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    dependencies = _build_analysis_dependencies(settings, llm)
    app_container = ApplicationContainer(
        analysis=AnalysisService(dependencies),
        audio_storage=LocalAudioStorage(settings.max_audio_bytes),
    )
    return app_container


def _build_analysis_dependencies(
    settings: Settings, llm: StructuredLLMPort
) -> AnalysisDependencies:
    t = Transcriber(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    return AnalysisDependencies(
        transcriber=t, diarizer=Diarizer(),
        classifier=ClassifierAgent(llm), quality=QualityAgent(llm),
        compliance=ComplianceAgent(llm), summarizer=SummarizerAgent(llm),
    )
