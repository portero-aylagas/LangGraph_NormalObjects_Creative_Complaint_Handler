import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import MODEL_NAME


def _llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to your environment or .env file."
        )
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _llm_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    response = _llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = _strip_code_fence(response.content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"LLM returned invalid JSON: {content}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM returned JSON that is not an object: {content}")
    return parsed
