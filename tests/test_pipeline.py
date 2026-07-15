"""Integration-style test for the supervisor with dependency doubles."""

from pathlib import Path

import pytest

from asr.diarizer import Diarizer
from models.schemas import (
    ClassificationResult,
    ComplianceResult,
    QualityChecklist,
    QualityResult,
    RawSegment,
    SummaryResult,
)
from pipeline import Pipeline
from services.analysis import AnalysisService


class FakeTranscriber:
    async def transcribe(self, audio_path):
        assert audio_path == Path("call.wav")
        return [
            RawSegment(start=0, end=2, text="Добрый день, МТБанк."),
            RawSegment(start=2, end=4, text="Нужна кредитная карта."),
        ]


class FakeClassifier:
    async def run(self, transcript):
        return ClassificationResult(topic="карты", priority="medium")


class FakeQuality:
    async def run(self, transcript):
        return QualityResult(
            total=50,
            checklist=QualityChecklist(
                greeting=True,
                need_detection=True,
                solution_provided=False,
                farewell=False,
            ),
        )


class FakeCompliance:
    async def run(self, transcript):
        return ComplianceResult(passed=True)


class FakeSummarizer:
    async def run(self, transcript, **peer_results):
        assert peer_results["classification"].topic == "карты"
        return SummaryResult(
            summary="Клиент запросил кредитную карту.",
            action_items=["Уточнить требования клиента"],
        )


@pytest.mark.asyncio
async def test_supervisor_builds_complete_response() -> None:
    service = AnalysisService(
        transcriber=FakeTranscriber(),
        diarizer=Diarizer(),
        classifier=FakeClassifier(),
        quality=FakeQuality(),
        compliance=FakeCompliance(),
        summarizer=FakeSummarizer(),
    )

    result = await service.analyze(Path("call.wav"))

    assert result.transcript[0].speaker == "Оператор"
    assert result.classification.topic == "карты"
    assert result.quality_score.total == 50
    assert result.summary == "Клиент запросил кредитную карту."
    assert "## Анализ звонка" in Pipeline._format_markdown(result)
