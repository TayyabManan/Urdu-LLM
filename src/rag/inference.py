"""Urdu RAG client — POSTs to the Modal /rag endpoint.

Local code no longer hosts bge-m3 or Qdrant (WSL OOM at 15GB cap). The Modal
endpoint does retrieve + generate server-side; we just POST a query.

Backwards-compatible API:
    pipeline = build_rag_pipeline()   # returns a tiny client config
    result = query_rag(pipeline, query, system=...)
    # result: {"answer": str, "sources": list[dict], "meta": dict}
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

DEFAULT_TOP_K = 3
DEFAULT_SYSTEM = (
    "You are a helpful assistant. Use the provided context to answer in Urdu. "
    "If the context is insufficient, answer briefly from general knowledge."
)


@dataclass
class RAGClient:
    rag_url: str
    top_k: int = DEFAULT_TOP_K
    max_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.3
    timeout_s: int = 180


def _derive_rag_url() -> str:
    """Convert FT_ENDPOINT_URL (…/generate) → …/rag, or use FT_RAG_URL directly."""
    explicit = os.environ.get("FT_RAG_URL")
    if explicit:
        return explicit
    gen = os.environ.get("FT_ENDPOINT_URL")
    if not gen:
        raise RuntimeError(
            "Set FT_RAG_URL or FT_ENDPOINT_URL in .env. "
            "Example: FT_ENDPOINT_URL=https://...-ftserver-generate.modal.run"
        )
    # Swap '-generate' → '-rag'
    if gen.endswith("-generate.modal.run"):
        return gen.replace("-generate.modal.run", "-rag.modal.run")
    if "-generate.modal.run/" in gen:
        return gen.replace("-generate.modal.run/", "-rag.modal.run/")
    raise RuntimeError(
        f"Cannot derive RAG URL from FT_ENDPOINT_URL={gen!r}. Set FT_RAG_URL explicitly."
    )


def build_rag_pipeline(
    top_k: int = DEFAULT_TOP_K,
    embed_device: str | None = None,  # unused — kept for signature compatibility
    ft_endpoint_url: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
    qdrant_path=None,  # unused — kept for compat
    collection: str = "urdu_wikipedia_v1",  # unused — Modal endpoint owns this
) -> RAGClient:
    rag_url = ft_endpoint_url or _derive_rag_url()
    return RAGClient(
        rag_url=rag_url,
        top_k=top_k,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def query_rag(client: RAGClient, query: str, system: str = DEFAULT_SYSTEM,
              use_adapter: bool = True, mode: str = "hybrid") -> dict:
    payload = {
        "query": query,
        "top_k": client.top_k,
        "max_tokens": client.max_tokens,
        "temperature": client.temperature,
        "top_p": client.top_p,
        "repetition_penalty": client.repetition_penalty,
        "use_adapter": use_adapter,
        "mode": mode,
    }
    # follow_redirects: a cold container returns Modal's 303 async-poll redirect
    # when the first request outlasts the sync window; following it returns the result.
    with httpx.Client(timeout=client.timeout_s, follow_redirects=True) as http:
        r = http.post(client.rag_url, json=payload)
        r.raise_for_status()
        data = r.json()
    return {
        "answer": data.get("answer", ""),
        "sources": data.get("sources", []),
        "meta": {
            "tokens_in": data.get("tokens_in"),
            "tokens_out": data.get("tokens_out"),
            "mode": data.get("mode"),
            "used_adapter": data.get("used_adapter", True),
        },
    }
