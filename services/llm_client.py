"""OpenAI-compatible structured-output client used by all agents."""

import json
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

ResultT = TypeVar("ResultT", bound=BaseModel)


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResultT],
    ) -> ResultT:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n"
                        "Верни только JSON, соответствующий этой JSON Schema:\n"
                        f"{schema}"
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("LLM вернула пустой ответ")
        return response_model.model_validate_json(content)
