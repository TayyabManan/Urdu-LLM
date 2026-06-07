"""Haystack component wrapping the Modal-served FT endpoint as a chat generator.

POSTs to scripts/19_serve_ft_endpoint.py's `generate` web function.
Endpoint URL via env var FT_ENDPOINT_URL or constructor arg.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from haystack import component


DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_RETRIES = 2


@component
class ModalFTChatGenerator:
    def __init__(
        self,
        endpoint_url: str | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        bearer_token: str | None = None,
    ):
        self.endpoint_url = endpoint_url or os.environ.get("FT_ENDPOINT_URL")
        if not self.endpoint_url:
            raise RuntimeError(
                "ModalFTChatGenerator: set FT_ENDPOINT_URL env var or pass endpoint_url."
            )
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.bearer_token = bearer_token or os.environ.get("FT_ENDPOINT_TOKEN")

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.bearer_token:
            h["Authorization"] = f"Bearer {self.bearer_token}"
        return h

    def _payload(self, prompt: str, system: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": 1.3,  # combat long-context Urdu collapse
        }
        if system:
            body["system"] = system
        return body

    async def _invoke(self, prompt: str, system: str | None) -> dict[str, Any]:
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    r = await client.post(
                        self.endpoint_url,
                        json=self._payload(prompt, system),
                        headers=self._headers(),
                    )
                    r.raise_for_status()
                    return r.json()
                except Exception as e:
                    last_err = e
                    if attempt < self.max_retries:
                        await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Modal FT endpoint failed after {self.max_retries + 1} tries: {last_err!r}")

    @component.output_types(replies=list[str], meta=list[dict])
    def run(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        result = asyncio.run(self._invoke(prompt, system))
        return {
            "replies": [result.get("text", "")],
            "meta": [{
                "tokens_in": result.get("tokens_in"),
                "tokens_out": result.get("tokens_out"),
            }],
        }

    @component.output_types(replies=list[str], meta=list[dict])
    async def run_async(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        result = await self._invoke(prompt, system)
        return {
            "replies": [result.get("text", "")],
            "meta": [{
                "tokens_in": result.get("tokens_in"),
                "tokens_out": result.get("tokens_out"),
            }],
        }
