"""Container exposes abstractions to HTTP and OpenWebUI entry points."""

from dataclasses import dataclass

from core.ports import AnalysisUseCase, AudioStoragePort


@dataclass(frozen=True)
class ApplicationContainer:
    analysis: AnalysisUseCase
    audio_storage: AudioStoragePort
