"""Supervisor pattern coordinating ASR and four specialized agents (LangChain LCEL)."""

import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.runnables import RunnableLambda, RunnableParallel

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
    ClassificationResult,
    ComplianceResult,
    QualityResult,
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
        self._agents = RunnableParallel(
            classification=RunnableLambda(dependencies.classifier.run),
            quality=RunnableLambda(dependencies.quality.run),
            compliance=RunnableLambda(dependencies.compliance.run),
            summary=RunnableLambda(dependencies.summarizer.run),
        )


    async def analyze(self, audio_path: Path) -> AnalysisResponse:
        log_event(self.logger, "analysis_started", audio=str(audio_path))
        raw = await self.dependencies.transcriber.transcribe(audio_path)
        transcript = await self.dependencies.diarizer.assign_speakers(raw)
        if not transcript:
            raise ValueError("В аудио не удалось распознать речь")
        results = await self._agents.ainvoke(transcript)
        response = self._build_response(transcript, **results)
        self._log_completed(response)
        return response


    @staticmethod
    def _build_response(
        transcript: list[TranscriptSegment],
        classification: ClassificationResult,
        quality: QualityResult,
        compliance: ComplianceResult,
        summary: SummaryResult,
    ) -> AnalysisResponse:
        return AnalysisResponse(
            transcript=transcript,
            classification=classification,
            quality_score=quality,
            compliance=compliance,
            summary=summary.summary,
            action_items=summary.action_items,
        )


    def _log_completed(self, response: AnalysisResponse) -> None:
        log_event(
            logger=self.logger,
            message="analysis_completed",
            transcript_segments=len(response.transcript),
            topic=response.classification.topic,
            quality_score=response.quality_score.total,
            compliance_passed=response.compliance.passed,
        )
