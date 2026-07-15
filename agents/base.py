"""Template Method shared by concrete contact-center agents."""

import logging
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar

from pydantic import BaseModel

from models.schemas import TranscriptSegment
from utils.logging import log_event

ResultT = TypeVar("ResultT", bound=BaseModel)


class StructuredLLM(Protocol):
    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResultT],
    ) -> ResultT: ...


class BaseAgent(ABC):
    name: str

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm
        self.logger = logging.getLogger(f"agent.{self.name}")

    async def _execute(
        self,
        transcript: list[TranscriptSegment],
        response_model: type[ResultT],
        extra_context: str = "",
    ) -> ResultT:
        transcript_text = self._format_transcript(transcript)
        log_event(
            self.logger,
            "agent_started",
            agent=self.name,
            transcript_segments=len(transcript),
        )
        try:
            result = await self.llm.complete_json(
                system_prompt=self.system_prompt(),
                user_prompt=f"{transcript_text}\n{extra_context}".strip(),
                response_model=response_model,
            )
        except Exception:
            self.logger.exception(
                "agent_failed",
                extra={"event_data": {"agent": self.name}},
            )
            raise
        log_event(
            self.logger,
            "agent_completed",
            agent=self.name,
            output=result.model_dump(mode="json"),
        )
        return result

    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError

    @staticmethod
    def _format_transcript(transcript: list[TranscriptSegment]) -> str:
        return "\n".join(
            f"[{item.start:.1f}-{item.end:.1f}] {item.speaker}: {item.text}"
            for item in transcript
        )
