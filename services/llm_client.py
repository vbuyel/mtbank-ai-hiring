"""OpenAI-compatible structured-output client used by all agents."""

import json
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from core.ports import StructuredLLMPort

ResultT = TypeVar("ResultT", bound=BaseModel)


class LLMClient(StructuredLLMPort):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model


    async def complete_json(
        self, system_prompt: str, user_prompt: str, response_model: type[ResultT]
    ) -> ResultT:
        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=self._messages(system_prompt, user_prompt, response_model),
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("LLM вернула пустой ответ")
        return response_model.model_validate_json(content)


    @staticmethod
    def _messages(system: str, user: str, model: type[BaseModel]) -> list[dict]:
        schema = json.dumps(model.model_json_schema(), ensure_ascii=False)
        instruction = (
            f"{system}\nВерни только JSON по этой JSON Schema:\n{schema}"
        )
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user},
        ]
        return messages
