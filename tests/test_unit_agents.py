"""Deterministic unit tests for every LLM-backed agent."""

from collections.abc import Callable

import pytest

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent
from asr.diarizer import Diarizer
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
    TranscriptSegment,
)


class FakeLLM:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    async def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def transcript() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            speaker="Оператор",
            start=0,
            end=2,
            text="Добрый день, чем могу помочь?",
        ),
        TranscriptSegment(
            speaker="Клиент",
            start=2,
            end=5,
            text="Хочу узнать условия кредита.",
        ),
    ]


@pytest.mark.asyncio
async def test_summarizer_forwards_formatted_transcript(transcript) -> None:
    expected = SummaryResult(summary="Клиент спросил про кредит.", action_items=[])
    llm = FakeLLM(expected)

    result = await SummarizerAgent(llm).run(transcript)

    assert result is expected
    assert llm.calls[0]["response_model"] is SummaryResult
    assert llm.calls[0]["user_prompt"] == (
        "[0.0-2.0] Оператор: Добрый день, чем могу помочь?\n"
        "[2.0-5.0] Клиент: Хочу узнать условия кредита."
    )
    assert "суммаризатор" in llm.calls[0]["system_prompt"]


@pytest.mark.asyncio
async def test_classifier_corrects_unsupported_complaint_topic(transcript) -> None:
    llm = FakeLLM(ClassificationResult(topic="жалобы", priority="medium"))

    result = await ClassifierAgent(llm).run(transcript)

    assert result == ClassificationResult(topic="кредиты", priority="medium")
    assert llm.calls[0]["response_model"] is ClassificationResult


@pytest.mark.asyncio
async def test_classifier_keeps_evidenced_complaint_topic() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Клиент",
            start=0,
            end=1,
            text="У меня жалоба на условия кредита.",
        )
    ]
    llm = FakeLLM(ClassificationResult(topic="жалобы", priority="high"))

    result = await ClassifierAgent(llm).run(transcript)

    assert result.topic == "жалобы"
    assert result.priority == "high"


@pytest.mark.asyncio
async def test_quality_recalculates_weighted_total(transcript) -> None:
    checklist = QualityChecklist(
        greeting=True,
        need_detection=False,
        solution_provided=True,
        farewell=True,
    )
    llm = FakeLLM(QualityResult(total=0, checklist=checklist))

    result = await QualityAgent(llm).run(transcript)

    assert result.total == 75
    assert result.checklist is checklist
    assert llm.calls[0]["response_model"] is QualityResult


@pytest.mark.asyncio
async def test_compliance_retains_only_operator_evidence(transcript) -> None:
    issues = [
        ComplianceIssue(
            rule="Гарантия",
            quote="«Добрый день, чем могу помочь?»",
            explanation="Проверяем нормализацию кавычек.",
        ),
        ComplianceIssue(
            rule="Чужая реплика",
            quote="Хочу узнать условия кредита.",
            explanation="Это сказал клиент.",
        ),
        ComplianceIssue(
            rule="Выдумка",
            quote="Мы гарантируем одобрение.",
            explanation="Такой реплики нет.",
        ),
    ]
    llm = FakeLLM(ComplianceResult(passed=True, issues=issues))

    result = await ComplianceAgent(llm).run(transcript)

    assert result.passed is False
    assert result.issues == [issues[0]]
    assert llm.calls[0]["response_model"] is ComplianceResult


@pytest.mark.asyncio
async def test_compliance_passes_when_all_evidence_is_invalid(transcript) -> None:
    issue = ComplianceIssue(
        rule="Выдумка",
        quote="",
        explanation="Пустая цитата.",
    )
    llm = FakeLLM(ComplianceResult(passed=False, issues=[issue]))

    result = await ComplianceAgent(llm).run(transcript)

    assert result == ComplianceResult(passed=True, issues=[])


AgentFactory = Callable[[FakeLLM], object]


@pytest.mark.parametrize(
    "factory",
    [ClassifierAgent, QualityAgent, ComplianceAgent, SummarizerAgent],
    ids=["classifier", "quality", "compliance", "summarizer"],
)
@pytest.mark.asyncio
async def test_agents_propagate_llm_errors(factory: AgentFactory, transcript) -> None:
    llm = FakeLLM(error=RuntimeError("LLM unavailable"))

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await factory(llm).run(transcript)

    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_diarizer_skips_llm_for_empty_input() -> None:
    llm = FakeLLM(error=AssertionError("LLM must not be called"))

    assert await Diarizer(llm).assign_speakers([]) == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_diarizer_sends_indexed_dialogue_and_schema_to_llm() -> None:
    raw = [
        RawSegment(start=0, end=1.5, text="Добрый день."),
        RawSegment(start=1.5, end=3, text="Расскажите об условиях."),
    ]
    llm = FakeLLM(
        DiarizationResult(
            speakers=[
                SegmentSpeaker(index=0, speaker="Оператор"),
                SegmentSpeaker(index=1, speaker="Клиент"),
            ]
        )
    )

    result = await Diarizer(llm).assign_speakers(raw)

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["response_model"] is DiarizationResult
    assert call["user_prompt"] == (
        "Разметь по ролям следующие реплики:\n"
        "0: Добрый день.\n"
        "1: Расскажите об условиях."
    )
    assert "банковского звонка МТБанк" in call["system_prompt"]
    assert "'Оператор' или 'Клиент'" in call["system_prompt"]
    assert [item.model_dump() for item in result] == [
        {
            "start": 0.0,
            "end": 1.5,
            "text": "Добрый день.",
            "speaker": "Оператор",
        },
        {
            "start": 1.5,
            "end": 3.0,
            "text": "Расскажите об условиях.",
            "speaker": "Клиент",
        },
    ]


@pytest.mark.asyncio
async def test_diarizer_maps_llm_response_by_index_with_client_fallback() -> None:
    raw = [
        RawSegment(start=0, end=1, text="Первая нейтральная реплика."),
        RawSegment(start=1, end=2, text="Вторая нейтральная реплика."),
        RawSegment(start=2, end=3, text="Третья нейтральная реплика."),
    ]
    llm = FakeLLM(
        DiarizationResult(
            speakers=[
                SegmentSpeaker(index=2, speaker="Оператор"),
                SegmentSpeaker(index=99, speaker="Оператор"),
            ]
        )
    )

    result = await Diarizer(llm).assign_speakers(raw)

    assert [item.speaker for item in result] == [
        "Клиент",
        "Клиент",
        "Оператор",
    ]


@pytest.mark.asyncio
async def test_diarizer_propagates_llm_errors() -> None:
    llm = FakeLLM(error=RuntimeError("LLM unavailable"))
    raw = [RawSegment(start=0, end=1, text="Нейтральная реплика.")]

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await Diarizer(llm).assign_speakers(raw)

    assert len(llm.calls) == 1
    assert llm.calls[0]["response_model"] is DiarizationResult
