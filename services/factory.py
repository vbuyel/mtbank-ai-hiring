"""Composition root: the only place that wires concrete dependencies."""

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent
from asr.diarizer import Diarizer
from asr.transcriber import Transcriber
from config import Settings
from services.analysis import AnalysisService
from services.llm_client import LLMClient


def build_analysis_service(settings: Settings) -> AnalysisService:
    llm = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    return AnalysisService(
        transcriber=Transcriber(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        ),
        diarizer=Diarizer(),
        classifier=ClassifierAgent(llm),
        quality=QualityAgent(llm),
        compliance=ComplianceAgent(llm),
        summarizer=SummarizerAgent(llm),
    )
