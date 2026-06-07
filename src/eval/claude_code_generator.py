"""Haystack component bridging Claude Code CLI as a chat generator.

Uses the local `claude` CLI (Claude Max subscription auth) via subprocess.
No Anthropic API key required.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from typing import Any

from haystack import component


@dataclass
class _ClaudeCLIResult:
    text: str
    model: str | None
    raw: dict[str, Any]


@component
class ClaudeCodeChatGenerator:
    """Calls `claude -p <prompt> --output-format json` and returns the assistant reply.

    The CLI uses your existing Claude Max login; verify with `claude --version`.
    Returns Haystack-shaped output: {"replies": [str], "meta": [dict]} to match
    other chat generators' interface enough for downstream parsing.
    """

    def __init__(self, timeout_s: int = 120, cli_path: str | None = None):
        self.cli_path = cli_path or shutil.which("claude")
        if not self.cli_path:
            raise RuntimeError(
                "claude CLI not found on PATH. Install Claude Code + run `claude login`."
            )
        self.timeout_s = timeout_s

    async def _invoke(self, prompt: str) -> _ClaudeCLIResult:
        proc = await asyncio.create_subprocess_exec(
            self.cli_path,
            "-p",
            prompt,
            "--output-format",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude CLI timed out after {self.timeout_s}s")

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI exit {proc.returncode}: {stderr.decode('utf-8', 'ignore')[:500]}"
            )

        raw_text = stdout.decode("utf-8", "ignore").strip()
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude CLI returned non-JSON: {raw_text[:300]}") from e

        text = raw.get("result") or raw.get("text") or ""
        model_usage = raw.get("modelUsage") or {}
        model = next(iter(model_usage.keys()), None) or raw.get("session_id")
        return _ClaudeCLIResult(text=str(text), model=model, raw=raw)

    @component.output_types(replies=list[str], meta=list[dict])
    def run(self, prompt: str) -> dict[str, Any]:
        result = asyncio.run(self._invoke(prompt))
        return {
            "replies": [result.text],
            "meta": [{"model": result.model, "raw": result.raw}],
        }

    @component.output_types(replies=list[str], meta=list[dict])
    async def run_async(self, prompt: str) -> dict[str, Any]:
        result = await self._invoke(prompt)
        return {
            "replies": [result.text],
            "meta": [{"model": result.model, "raw": result.raw}],
        }
