"""Judge generator factory.

One generator per judge: OpenAI, Google Gemini, Claude Code.
Uses Haystack chat generators directly (no Pipeline wrapper — one-step calls
don't need DAG orchestration; Pipeline pays off in the upcoming RAG layer).

All three judges receive the SAME rendered prompt (from prompts/judge_v2.j2)
and return one JSON line per eval item.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from src.eval.claude_code_generator import ClaudeCodeChatGenerator


JUDGE_DIR_NAMES = {
    "openai": "openai-api-gpt-5.3-chat-latest",
    "gemini": "google-api-gemini-2.5-flash",
    "claude": "claude-code-opus-4.7-auto",
}


@dataclass
class JudgeRunner:
    name: str
    out_dir_name: str
    model_label: str
    call: Callable[[str], Any]


def _build_openai_runner() -> JudgeRunner:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env.")
    gen = OpenAIChatGenerator(
        model="gpt-5.3-chat-latest",
        api_key=Secret.from_token(key),
        generation_kwargs={"response_format": {"type": "json_object"}},
    )

    def call(prompt: str) -> dict[str, Any]:
        result = gen.run(messages=[ChatMessage.from_user(prompt)])
        reply = result["replies"][0]
        text = reply.text if hasattr(reply, "text") else str(reply)
        return {"text": text, "model": "gpt-5.3-chat-latest"}

    return JudgeRunner(
        name="openai",
        out_dir_name=JUDGE_DIR_NAMES["openai"],
        model_label="gpt-5.3-chat-latest",
        call=call,
    )


def _build_gemini_runner() -> JudgeRunner:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY missing. Set it in .env.")
    # Imported lazily to avoid hard dep when only running other judges
    from haystack_integrations.components.generators.google_genai import (
        GoogleGenAIChatGenerator,
    )

    gen = GoogleGenAIChatGenerator(
        model="gemini-2.5-flash",
        api_key=Secret.from_token(key),
        generation_kwargs={"response_mime_type": "application/json"},
    )

    def call(prompt: str) -> dict[str, Any]:
        result = gen.run(messages=[ChatMessage.from_user(prompt)])
        reply = result["replies"][0]
        text = reply.text if hasattr(reply, "text") else str(reply)
        return {"text": text, "model": "gemini-2.5-flash"}

    return JudgeRunner(
        name="gemini",
        out_dir_name=JUDGE_DIR_NAMES["gemini"],
        model_label="gemini-2.5-flash",
        call=call,
    )


def _build_claude_runner() -> JudgeRunner:
    gen = ClaudeCodeChatGenerator()

    def call(prompt: str) -> dict[str, Any]:
        result = gen.run(prompt=prompt)
        text = result["replies"][0]
        model = result["meta"][0].get("model") or "claude-code"
        return {"text": text, "model": model}

    return JudgeRunner(
        name="claude",
        out_dir_name=JUDGE_DIR_NAMES["claude"],
        model_label="claude-code",
        call=call,
    )


_BUILDERS = {
    "openai": _build_openai_runner,
    "gemini": _build_gemini_runner,
    "claude": _build_claude_runner,
}


def build_judge_runner(name: str) -> JudgeRunner:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown judge {name!r}. Pick from {list(_BUILDERS)}.")
    return _BUILDERS[name]()


def load_judge_template() -> str:
    template_path = Path(__file__).parent.parent.parent / "prompts" / "judge_v2.j2"
    if not template_path.exists():
        raise RuntimeError(f"Judge template not found at {template_path}")
    return template_path.read_text(encoding="utf-8")
