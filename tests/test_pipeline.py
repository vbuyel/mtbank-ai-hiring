"""Integration-style test for the supervisor with dependency doubles."""

import asyncio
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
    async def run(self, transcript):
        assert transcript[1].text == "Нужна кредитная карта."
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
async def test_pipeline_skips_open_webui_background_tasks() -> None:
    pipeline = object.__new__(Pipeline)
    body = await pipeline.inlet(
        {"metadata": {"task": "title_generation"}},
    )
    body.pop("metadata")

    result = pipeline.pipe(
        user_message="Фоновая задача",
        model_id="mtbank",
        messages=[],
        body=body,
    )

    assert body["task"] == "title_generation"
    assert body["skip_audio"] is True
    assert result == "Фоновая задача"


@pytest.mark.asyncio
async def test_supervisor_builds_complete_response() -> None:
    service = build_service()
    result = await service.analyze(Path("call.wav"))
    assert result.transcript[0].speaker == "Оператор"
    assert result.classification.topic == "карты"
    assert result.quality_score.total == 50
    assert result.summary == "Клиент запросил кредитную карту."
    assert "## Анализ звонка" in Pipeline._format_markdown(result)


@pytest.mark.asyncio
async def test_supervisor_runs_all_agents_in_parallel() -> None:
    all_started = asyncio.Event()
    started = 0

    async def wait_for_peers() -> None:
        nonlocal started
        started += 1
        if started == 4:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1)

    class ConcurrentClassifier:
        async def run(self, transcript):
            await wait_for_peers()
            return ClassificationResult(topic="карты", priority="medium")

    class ConcurrentQuality:
        async def run(self, transcript):
            await wait_for_peers()
            return QualityResult(
                total=50,
                checklist=QualityChecklist(
                    greeting=True,
                    need_detection=True,
                    solution_provided=False,
                    farewell=False,
                ),
            )

    class ConcurrentCompliance:
        async def run(self, transcript):
            await wait_for_peers()
            return ComplianceResult(passed=True)

    class ConcurrentSummarizer:
        async def run(self, transcript):
            await wait_for_peers()
            return SummaryResult(summary="Резюме")

    service = AnalysisService(
        AnalysisDependencies(
            transcriber=FakeTranscriber(),
            diarizer=FakeDiarizer(),
            classifier=ConcurrentClassifier(),
            quality=ConcurrentQuality(),
            compliance=ConcurrentCompliance(),
            summarizer=ConcurrentSummarizer(),
        )
    )

    await service.analyze(Path("call.wav"))

    assert started == 4
