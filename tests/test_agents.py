"""Contract tests for the four agents without a real LLM."""

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

FAKE_RESPONSES = {
    ClassificationResult: ClassificationResult(topic="жалобы", priority="medium"),
    QualityResult: QualityResult(
        total=75,
        checklist=QualityChecklist(
            greeting=True, need_detection=True,
            solution_provided=True, farewell=False,
        ),
    ),
    ComplianceResult: ComplianceResult(
        passed=False,
        issues=[
            ComplianceIssue(
                rule="Гарантированная доходность",
                quote="",
                explanation="Цитата отсутствует",
            )
        ],
    ),
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


@pytest.mark.asyncio
async def test_classifier_returns_topic(transcript) -> None:
    llm = FakeLLMClient()
    classification = await ClassifierAgent(llm).run(transcript)

    assert classification.topic == "кредиты"


@pytest.mark.asyncio
async def test_quality_returns_score(transcript) -> None:
    llm = FakeLLMClient()
    quality = await QualityAgent(llm).run(transcript)

    assert quality.total == 85


@pytest.mark.asyncio
async def test_compliance_returns_status(transcript) -> None:
    llm = FakeLLMClient()
    compliance = await ComplianceAgent(llm).run(transcript)

    assert compliance.passed is True
    assert compliance.issues == []


@pytest.mark.asyncio
async def test_diarizer_corrects_clear_client_intent() -> None:
    class DiarizationLLM:
        def __init__(self):
            self.call = 0

        async def complete_json(self, system_prompt, user_prompt, response_model):
            del system_prompt, user_prompt, response_model
            self.call += 1
            if self.call == 2:
                return DiarizationResult(
                    speakers=[
                        SegmentSpeaker(index=0, speaker="Оператор"),
                        SegmentSpeaker(index=1, speaker="Клиент"),
                        SegmentSpeaker(index=2, speaker="Оператор"),
                        SegmentSpeaker(index=3, speaker="Оператор"),
                    ]
                )
            return DiarizationResult(
                speakers=[
                    SegmentSpeaker(index=0, speaker="Клиент"),
                    SegmentSpeaker(index=1, speaker="Оператор"),
                    SegmentSpeaker(index=2, speaker="Клиент"),
                    SegmentSpeaker(index=3, speaker="Клиент"),
                ]
            )

    segments = [
        RawSegment(start=0, end=1, text="Какая сумма вас интересует?"),
        RawSegment(start=1, end=2, text="Лучше онлайн."),
        RawSegment(start=2, end=3, text="Подскажите ваш имейл."),
        RawSegment(start=3, end=4, text="Есть еще вопросы?"),
    ]
    result = await Diarizer(DiarizationLLM()).assign_speakers(segments)
    assert [item.speaker for item in result] == [
        "Оператор", "Клиент", "Оператор", "Оператор"
    ]


@pytest.mark.asyncio
async def test_summarizer_uses_transcript_only(transcript) -> None:
    llm = FakeLLMClient()
    summary = await SummarizerAgent(llm).run(transcript)

    assert summary.action_items == ["Отправить клиенту условия"]
