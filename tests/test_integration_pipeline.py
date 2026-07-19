"""Pipeline integration tests with network and ASR boundaries replaced by fakes."""

from collections import Counter
from pathlib import Path

import pytest

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent
from asr.diarizer import Diarizer
from core.container import ApplicationContainer
from models.schemas import (
    ClassificationResult,
    ComplianceIssue,
    ComplianceResult,
    DiarizationResult,
    QualityChecklist,
    QualityResult,
    RawSegment,
    SegmentSpeaker,
    SummaryResult,
)
from pipeline import Pipeline
from services.analysis import AnalysisDependencies, AnalysisService


class FakeTranscriber:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    async def transcribe(self, audio_path: Path) -> list[RawSegment]:
        self.paths.append(audio_path)
        return [
            RawSegment(start=0, end=2, text="Добрый день, чем могу помочь?"),
            RawSegment(start=2, end=4, text="Хочу узнать условия кредита."),
        ]


class ModelAwareLLM:
    def __init__(self) -> None:
        self.response_models: list[type] = []

    async def complete_json(self, system_prompt, user_prompt, response_model):
        del system_prompt, user_prompt
        self.response_models.append(response_model)
        return self._responses()[response_model]

    @staticmethod
    def _responses() -> dict[type, object]:
        valid_issue = ComplianceIssue(
            rule="Проверка цитаты",
            quote="Добрый день, чем могу помочь?",
            explanation="Точная реплика оператора.",
        )
        invalid_issue = ComplianceIssue(
            rule="Выдуманная гарантия",
            quote="Кредит точно одобрен.",
            explanation="Этой реплики нет.",
        )
        return {
            DiarizationResult: DiarizationResult(
                speakers=[
                    SegmentSpeaker(index=0, speaker="Оператор"),
                    SegmentSpeaker(index=1, speaker="Клиент"),
                ]
            ),
            ClassificationResult: ClassificationResult(
                topic="жалобы",
                priority="medium",
            ),
            QualityResult: QualityResult(
                total=0,
                checklist=QualityChecklist(
                    greeting=True,
                    need_detection=True,
                    solution_provided=False,
                    farewell=False,
                ),
            ),
            ComplianceResult: ComplianceResult(
                passed=True,
                issues=[valid_issue, invalid_issue],
            ),
            SummaryResult: SummaryResult(
                summary="Клиент запросил условия кредита.",
                action_items=["Отправить условия"],
            ),
        }


class UnexpectedAudioStorage:
    async def from_url(self, url: str):
        raise AssertionError(f"Unexpected remote download: {url}")


def build_real_pipeline() -> tuple[Pipeline, FakeTranscriber, ModelAwareLLM]:
    transcriber = FakeTranscriber()
    llm = ModelAwareLLM()
    dependencies = AnalysisDependencies(
        transcriber=transcriber,
        diarizer=Diarizer(llm),
        classifier=ClassifierAgent(llm),
        quality=QualityAgent(llm),
        compliance=ComplianceAgent(llm),
        summarizer=SummarizerAgent(llm),
    )
    pipeline = Pipeline()
    pipeline.container = ApplicationContainer(
        analysis=AnalysisService(dependencies),
        audio_storage=UnexpectedAudioStorage(),
    )
    return pipeline, transcriber, llm


def test_pipeline_runs_complete_analysis_with_boundary_fakes() -> None:
    pipeline, transcriber, llm = build_real_pipeline()

    try:
        result = pipeline.pipe("ignored", "mtbank", [], {"audio_url": "call.wav"})
    finally:
        pipeline._runner.close()

    assert transcriber.paths == [Path("call.wav")]
    assert "**Тема:** кредиты" in result
    assert "**Качество:** 50/100" in result
    assert "Проверка цитаты" in result
    assert "Выдуманная гарантия" not in result
    assert "Клиент запросил условия кредита." in result
    assert "- Отправить условия" in result
    assert Counter(llm.response_models) == Counter(
        {
            DiarizationResult: 1,
            ClassificationResult: 1,
            QualityResult: 1,
            ComplianceResult: 1,
            SummaryResult: 1,
        }
    )


class TrackingAudio:
    def __init__(self) -> None:
        self.closed = False

    @property
    def path(self) -> Path:
        return Path("downloaded.wav")

    def close(self) -> None:
        self.closed = True


class FakeAudioStorage:
    def __init__(self, audio: TrackingAudio) -> None:
        self.audio = audio
        self.urls: list[str] = []

    async def from_url(self, url: str) -> TrackingAudio:
        self.urls.append(url)
        return self.audio


class FailingAnalysis:
    async def analyze(self, audio_path: Path):
        assert audio_path == Path("downloaded.wav")
        raise RuntimeError("analysis failed")


def test_pipeline_closes_downloaded_audio_when_analysis_fails() -> None:
    audio = TrackingAudio()
    storage = FakeAudioStorage(audio)
    pipeline = Pipeline()
    pipeline.container = ApplicationContainer(
        analysis=FailingAnalysis(),
        audio_storage=storage,
    )

    try:
        with pytest.raises(RuntimeError, match="analysis failed"):
            pipeline.pipe(
                "ignored",
                "mtbank",
                [],
                {"audio_url": "https://example.test/call.wav"},
            )
    finally:
        pipeline._runner.close()

    assert storage.urls == ["https://example.test/call.wav"]
    assert audio.closed is True
