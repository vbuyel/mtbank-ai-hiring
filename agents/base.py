"""Template Method shared by concrete contact-center agents."""

import logging
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from core.ports import StructuredLLMPort
from models.schemas import TranscriptSegment
from utils.logging import log_event

ResultT = TypeVar("ResultT", bound=BaseModel)


class BaseAgent(ABC):
    name: str


    def __init__(self, llm: StructuredLLMPort) -> None:
        self.llm = llm
        self.logger = logging.getLogger(f"agent.{self.name}")


    async def _execute(
        self,
        transcript: list[TranscriptSegment],
        response_model: type[ResultT],
        extra_context: str = "",
    ) -> ResultT:
        self._log_started(transcript)
        user_prompt = self._build_user_prompt(transcript, extra_context)
        try:
            result = await self.llm.complete_json(
                system_prompt=self.system_prompt(),
                user_prompt=user_prompt,
                response_model=response_model,
            )
        except Exception:
            self._log_failed()
            raise
        self._log_completed(result)
        return result


    @abstractmethod
    def system_prompt(self) -> str:
        pass


    def _log_started(self, transcript: list[TranscriptSegment]) -> None:
        log_event(
            self.logger,
            "agent_started",
            agent=self.name,
            transcript_segments=len(transcript),
        )


    def _log_failed(self) -> None:
        self.logger.exception(
            "agent_failed",
            extra={"event_data": {"agent": self.name}},
        )


    def _log_completed(self, result: BaseModel) -> None:
        log_event(
            self.logger,
            "agent_completed",
            agent=self.name,
            output=result.model_dump(mode="json"),
        )


    def _build_user_prompt(
        self,
        transcript: list[TranscriptSegment],
        context: str,
    ) -> str:
        formatted_transcript = self._format_transcript(transcript)
        return f"{formatted_transcript}\n{context}".strip()


    @staticmethod
    def _format_transcript(transcript: list[TranscriptSegment]) -> str:
        formatted_transcript = [
            f"[{item.start:.1f}-{item.end:.1f}] {item.speaker}: {item.text}"
            for item in transcript
        ]
        return "\n".join(formatted_transcript)
