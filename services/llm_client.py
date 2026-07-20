"""OpenAI-compatible structured-output client used by all agents."""

import asyncio
import json
import logging
from typing import TypeVar
import re

from openai import AsyncOpenAI
from pydantic import BaseModel

from core.ports import StructuredLLMPort

ResultT = TypeVar("ResultT", bound=BaseModel)
log = logging.getLogger(__name__)


class LLMClient(StructuredLLMPort):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model


    async def complete_json(
        self, system_prompt: str, user_prompt: str, response_model: type[ResultT]
    ) -> ResultT:
        last_error = None
        for attempt in range(3):
            try:
                return await self._attempt(system_prompt, user_prompt, response_model)
            except ValueError as e:
                last_error = e
                wait = 2 ** attempt
                log.warning("LLM attempt %d failed: %s — retrying in %ds", attempt + 1, e, wait)
                await asyncio.sleep(wait)
        raise last_error


    async def _attempt(
        self, system_prompt: str, user_prompt: str, response_model: type[ResultT]
    ) -> ResultT:
        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=self._messages(system_prompt, user_prompt, response_model),
        )
        if not completion.choices:
            raise ValueError("LLM вернул пустой список choices")
        message = completion.choices[0].message
        if message is None or message.content is None:
            raise ValueError("LLM вернул ответ без content")
        return response_model.model_validate_json(self._extract_json(message.content))


    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from LLM response that may contain markdown fences or prose."""
        stripped = text.strip()
        if stripped.startswith("{"):
            return stripped
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
        start = stripped.find("{")
        end = stripped.rfind("}") + 1
        if start != -1 and end > start:
            return stripped[start:end]
        return stripped


    @staticmethod
    def _messages(system: str, user: str, model: type[BaseModel]) -> list[dict]:
        schema = json.dumps(model.model_json_schema(), ensure_ascii=False)
        instruction = (
            f"{system}\n\n"
            "You MUST return a JSON object conforming to the JSON Schema below. "
            "Do NOT return the schema itself. Do NOT nest your response under a 'properties' key. "
            "Extract/analyze the text and fill the fields with actual data.\n\n"
            f"JSON Schema:\n{schema}"
        )
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user},
        ]
        return messages
