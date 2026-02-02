"""
AI assistant utilities for summarizing dashboard data with OpenAI.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
import os

from openai import OpenAI


@dataclass
class AIPayload:
    snapshot_json: str
    prompt: str


def build_snapshot_payload(snapshot: Dict[str, Any], prompt: str) -> AIPayload:
    snapshot_json = json.dumps(snapshot, default=str)
    return AIPayload(snapshot_json=snapshot_json, prompt=prompt)


def run_ai_assistant(
    payload: AIPayload,
    model: str,
    include_web: bool,
    temperature: float = 0.2,
    max_output_tokens: int = 900,
) -> Tuple[Optional[str], Optional[str]]:
    if not os.getenv("OPENAI_API_KEY"):
        return None, "Missing OPENAI_API_KEY environment variable."

    client = OpenAI()
    tools = [{"type": "web_search_preview"}] if include_web else None

    system_prompt = (
        "You are a financial decision-support assistant for a retail investor. "
        "Use only the provided dashboard data unless web search is enabled. "
        "Provide concise, structured insights, highlight risks, and suggest "
        "risk-managed next steps (diversification, sizing, risk limits). "
        "Do not provide direct trade instructions, price targets, or guarantees. "
        "Always include a brief disclaimer that this is not financial advice."
    )

    user_text = (
        "Dashboard snapshot (JSON):\n"
        f"{payload.snapshot_json}\n\n"
        f"User request: {payload.prompt}\n"
        "Return: summary, key risks, opportunities, and a short action checklist."
    )

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                },
            ],
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        return None, f"OpenAI API error: {exc}"

    return response.output_text, None
