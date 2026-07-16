"""Integration-style test for the supervisor with dependency doubles."""

from pathlib import Path

import pytest

from models.schemas import (
    ClassificationResult,
    ComplianceResult,
    QualityChecklist,
    QualityResult,
    RawSegment,
    SummaryResult,
    TranscriptSegment,
)
from pipeline import Pipeline
from services.analysis import AnalysisDependencies, AnalysisService


class FakeDiarizer:
    async def assign_speakers(
        self, segments: list[RawSegment]
    ) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                speaker="Оператор" if i == 0 else "Клиент",
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
            for i, seg in enumerate(segments)
        ]


class FakeTranscriber:
    async def transcribe(self, audio_path):
        assert audio_path == Path("call.wav")
        raw_segments = [
            RawSegment(start=0, end=2, text="Добрый день, МТБанк."),
            RawSegment(start=2, end=4, text="Нужна кредитная карта."),
        ]
        return raw_segments


class FakeClassifier:
    async def run(self, transcript):
        return ClassificationResult(topic="карты", priority="medium")


class FakeQuality:
    async def run(self, transcript):
        checklist = QualityChecklist(
            greeting=True,
            need_detection=True,
            solution_provided=False,
            farewell=False,
        )
        result = QualityResult(
            total=50,
            checklist=checklist,
        )
        return result


class FakeCompliance:
    async def run(self, transcript):
        return ComplianceResult(passed=True)


class FakeSummarizer:
    async def run(self, transcript, context):
        assert context.classification.topic == "карты"
        result = SummaryResult(
            summary="Клиент запросил кредитную карту.",
            action_items=["Уточнить требования клиента"],
        )
        return result


def build_service() -> AnalysisService:
    dependencies = AnalysisDependencies(
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
        classifier=FakeClassifier(),
        quality=FakeQuality(),
        compliance=FakeCompliance(),
        summarizer=FakeSummarizer(),
    )
    analysis_service = AnalysisService(dependencies)
    return analysis_service


@pytest.mark.asyncio
async def test_supervisor_builds_complete_response() -> None:
    service = build_service()
    result = await service.analyze(Path("call.wav"))
    assert result.transcript[0].speaker == "Оператор"
    assert result.classification.topic == "карты"
    assert result.quality_score.total == 50
    assert result.summary == "Клиент запросил кредитную карту."
    assert "## Анализ звонка" in Pipeline._format_markdown(result)
