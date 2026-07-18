"""Abstract ports; outer adapters implement these application contracts."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from models.schemas import (
    AnalysisResponse,
    ClassificationResult,
    ComplianceResult,
    QualityResult,
    RawSegment,
    SummaryResult,
    TranscriptSegment,
)

ResultT = TypeVar("ResultT", bound=BaseModel)


class AnalysisUseCase(ABC):
    @abstractmethod
    async def analyze(self, audio_path: Path) -> AnalysisResponse:
        pass


class TranscriberPort(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: Path) -> list[RawSegment]:
        pass


class DiarizerPort(ABC):
    @abstractmethod
    async def assign_speakers(
        self, segments: list[RawSegment]
    ) -> list[TranscriptSegment]:
        pass


class ClassifierPort(ABC):
    @abstractmethod
    async def run(
        self, transcript: list[TranscriptSegment]
    ) -> ClassificationResult:
        pass


class QualityPort(ABC):
    @abstractmethod
    async def run(
        self, transcript: list[TranscriptSegment]
    ) -> QualityResult:
        pass


class CompliancePort(ABC):
    @abstractmethod
    async def run(
        self, transcript: list[TranscriptSegment]
    ) -> ComplianceResult:
        pass


class SummarizerPort(ABC):
    @abstractmethod
    async def run(
        self, transcript: list[TranscriptSegment]
    ) -> SummaryResult:
        pass


class StructuredLLMPort(ABC):
    @abstractmethod
    async def complete_json(
        self, system_prompt: str, user_prompt: str, response_model: type[ResultT]
    ) -> ResultT:
        pass


class AudioResource(ABC):
    @property
    @abstractmethod
    def path(self) -> Path:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class AudioStoragePort(ABC):
    @abstractmethod
    async def from_upload(self, file: object) -> AudioResource:
        pass

    @abstractmethod
    async def from_url(self, url: str) -> AudioResource:
        pass
