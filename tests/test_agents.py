"""Contract tests for the four agents without a real LLM."""

import pytest

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent
from models.schemas import (
    ClassificationResult,
    ComplianceResult,
    QualityChecklist,
    QualityResult,
    SummaryContext,
    SummaryResult,
    TranscriptSegment,
)

FAKE_RESPONSES = {
    ClassificationResult: ClassificationResult(topic="кредиты", priority="medium"),
    QualityResult: QualityResult(
        total=75,
        checklist=QualityChecklist(
            greeting=True, need_detection=True,
            solution_provided=True, farewell=False,
        ),
    ),
    ComplianceResult: ComplianceResult(passed=True),
    SummaryResult: SummaryResult(
        summary="Клиент уточнил условия кредита.",
        action_items=["Отправить клиенту условия"],
    ),
}


class FakeLLMClient:
    async def complete_json(self, system_prompt, user_prompt, response_model):
        del system_prompt, user_prompt
        return FAKE_RESPONSES[response_model]


@pytest.fixture
def transcript() -> list[TranscriptSegment]:
    transcriptions = [
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
    return transcriptions


@pytest.fixture
def summary_context() -> SummaryContext:
    checklist = QualityChecklist(
        greeting=True, need_detection=True,
        solution_provided=True, farewell=False,
    )
    summary = SummaryContext(
        classification=ClassificationResult(topic="кредиты", priority="medium"),
        quality=QualityResult(total=75, checklist=checklist),
        compliance=ComplianceResult(passed=True),
    )
    return summary


@pytest.mark.asyncio
async def test_classifier_returns_topic(transcript) -> None:
    llm = FakeLLMClient()
    classification = await ClassifierAgent(llm).run(transcript)

    assert classification.topic == "кредиты"


@pytest.mark.asyncio
async def test_quality_returns_score(transcript) -> None:
    llm = FakeLLMClient()
    quality = await QualityAgent(llm).run(transcript)

    assert quality.total == 75


@pytest.mark.asyncio
async def test_compliance_returns_status(transcript) -> None:
    llm = FakeLLMClient()
    compliance = await ComplianceAgent(llm).run(transcript)

    assert compliance.passed is True


@pytest.mark.asyncio
async def test_summarizer_uses_peer_results(transcript, summary_context) -> None:
    llm = FakeLLMClient()
    summary = await SummarizerAgent(llm).run(transcript, summary_context)

    assert summary.action_items == ["Отправить клиенту условия"]
