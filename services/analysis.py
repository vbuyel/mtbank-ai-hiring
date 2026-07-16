"""Supervisor pattern coordinating ASR and four specialized agents."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from core.ports import (
    AnalysisUseCase,
    ClassifierPort,
    CompliancePort,
    DiarizerPort,
    QualityPort,
    SummarizerPort,
    TranscriberPort,
)
from models.schemas import (
    AnalysisResponse,
    SummaryContext,
    SummaryResult,
    TranscriptSegment,
)
from shared.logging import log_event


@dataclass(frozen=True)
class AnalysisDependencies:
    transcriber: TranscriberPort
    diarizer: DiarizerPort
    classifier: ClassifierPort
    quality: QualityPort
    compliance: CompliancePort
    summarizer: SummarizerPort


class AnalysisService(AnalysisUseCase):
    def __init__(self, dependencies: AnalysisDependencies) -> None:
        self.dependencies = dependencies
        self.logger = logging.getLogger("services.analysis")


    async def analyze(self, audio_path: Path) -> AnalysisResponse:
        log_event(self.logger, "analysis_started", audio=str(audio_path))
        raw = await self.dependencies.transcriber.transcribe(audio_path)
        transcript = await self.dependencies.diarizer.assign_speakers(raw)
        if not transcript:
            raise ValueError("В аудио не удалось распознать речь")
        context = await self._run_reviews(transcript)
        summary = await self.dependencies.summarizer.run(transcript, context)
        response = self._build_response(transcript, context, summary)
        self._log_completed(response)
        return response


    async def _run_reviews(
        self,
        transcript: list[TranscriptSegment],
    ) -> SummaryContext:
        classification, quality, compliance = await asyncio.gather(
            self.dependencies.classifier.run(transcript),
            self.dependencies.quality.run(transcript),
            self.dependencies.compliance.run(transcript),
        )
        summary = SummaryContext(
            classification=classification,
            quality=quality,
            compliance=compliance,
        )
        return summary


    @staticmethod
    def _build_response(
        transcript: list[TranscriptSegment],
        context: SummaryContext,
        summary: SummaryResult,
    ) -> AnalysisResponse:
        response = AnalysisResponse(
            transcript=transcript,
            classification=context.classification,
            quality_score=context.quality,
            compliance=context.compliance,
            summary=summary.summary,
            action_items=summary.action_items,
        )
        return response


    def _log_completed(self, response: AnalysisResponse) -> None:
        log_event(
            logger=self.logger,
            message="analysis_completed",
            transcript_segments=len(response.transcript),
            topic=response.classification.topic,
            quality_score=response.quality_score.total,
            compliance_passed=response.compliance.passed,
        )
