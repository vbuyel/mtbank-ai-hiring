"""Typed contracts shared by ASR, agents, API, and OpenWebUI."""

from typing import Literal

from pydantic import BaseModel, Field

Speaker = Literal["Оператор", "Клиент"]


class RawSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)


class TranscriptSegment(RawSegment):
    speaker: Speaker


class ClassificationResult(BaseModel):
    topic: Literal["кредиты", "карты", "переводы", "жалобы", "другое"]
    priority: Literal["low", "medium", "high"]


class SegmentSpeaker(BaseModel):
    index: int
    speaker: Speaker


class DiarizationResult(BaseModel):
    speakers: list[SegmentSpeaker]


class QualityChecklist(BaseModel):
    greeting: bool
    need_detection: bool
    solution_provided: bool
    farewell: bool


class QualityResult(BaseModel):
    total: int = Field(ge=0, le=100)
    checklist: QualityChecklist
    comment: str = ""


class ComplianceIssue(BaseModel):
    rule: str
    quote: str
    explanation: str


class ComplianceResult(BaseModel):
    passed: bool
    issues: list[ComplianceIssue] = Field(default_factory=list)


class SummaryResult(BaseModel):
    summary: str = Field(min_length=1)
    action_items: list[str] = Field(default_factory=list)


class SummaryContext(BaseModel):
    classification: ClassificationResult
    quality: QualityResult
    compliance: ComplianceResult


class AnalysisResponse(BaseModel):
    transcript: list[TranscriptSegment]
    classification: ClassificationResult
    quality_score: QualityResult
    compliance: ComplianceResult
    summary: str
    action_items: list[str]
