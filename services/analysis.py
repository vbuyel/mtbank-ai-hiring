"""Supervisor pattern coordinating ASR and four specialized agents."""

import asyncio
import logging
from pathlib import Path

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent
from asr.diarizer import Diarizer
from asr.transcriber import Transcriber
from models.schemas import AnalysisResponse
from utils.logging import log_event


class AnalysisService:
    def __init__(
        self,
        *,
        transcriber: Transcriber,
        diarizer: Diarizer,
        classifier: ClassifierAgent,
        quality: QualityAgent,
        compliance: ComplianceAgent,
        summarizer: SummarizerAgent,
    ) -> None:
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.classifier = classifier
        self.quality = quality
        self.compliance = compliance
        self.summarizer = summarizer
        self.logger = logging.getLogger("services.analysis")

    async def analyze(self, audio_path: str | Path) -> AnalysisResponse:
        log_event(self.logger, "analysis_started", audio=str(audio_path))
        raw_segments = await self.transcriber.transcribe(audio_path)
        transcript = self.diarizer.assign_speakers(raw_segments)
        if not transcript:
            raise ValueError("В аудио не удалось распознать речь")

        classification, quality, compliance = await asyncio.gather(
            self.classifier.run(transcript),
            self.quality.run(transcript),
            self.compliance.run(transcript),
        )
        summary = await self.summarizer.run(
            transcript,
            classification=classification,
            quality=quality,
            compliance=compliance,
        )

        response = AnalysisResponse(
            transcript=transcript,
            classification=classification,
            quality_score=quality,
            compliance=compliance,
            summary=summary.summary,
            action_items=summary.action_items,
        )
        log_event(
            self.logger,
            "analysis_completed",
            transcript_segments=len(transcript),
            topic=classification.topic,
            quality_score=quality.total,
            compliance_passed=compliance.passed,
        )
        return response
