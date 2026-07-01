"""
Canonical RAG surface form — the single source of truth for what the model
sees at RAG inference time.

v3's whole reason for existing is that v2 was trained on plain Q->A pairs and
never saw the `{context}\n\nسوال: {query}` shape the /rag endpoint serves, so
retrieved context pushed it out-of-distribution. To fix that, the v3 RAG
training triples must reproduce the inference format BYTE-FOR-BYTE.

These constants MUST stay identical to scripts/19_serve_ft_endpoint.py
(RAG_SYSTEM / RAG_TEMPLATE). That file defines its own copies because it runs
inside a Modal image and can't import this module; keep the two in sync. The
generator (scripts/23) and the formatter (scripts/26) both import from HERE so
they can never drift from each other.
"""
from __future__ import annotations

# System message used for every RAG turn (endpoint line ~76).
RAG_SYSTEM = (
    "You are a helpful assistant. Use the provided context to answer in Urdu. "
    "If the context is insufficient, answer briefly from general knowledge."
)

# User-message template (endpoint line ~69). "سوال:" is the literal Urdu for
# "question:". Chunks are joined with a blank line.
RAG_TEMPLATE = "{chunks}\n\nسوال: {query}"

CHUNK_SEPARATOR = "\n\n"


def build_rag_user(chunks: list[str], query: str) -> str:
    """Assemble the user message exactly as the /rag endpoint does.

    Mirrors scripts/19_serve_ft_endpoint.py:
        chunks_text = "\\n\\n".join(s["content"] for s in sources if s.get("content"))
        rag_prompt = RAG_TEMPLATE.format(chunks=chunks_text, query=query)
    """
    chunks_text = CHUNK_SEPARATOR.join(c for c in chunks if c and c.strip())
    return RAG_TEMPLATE.format(chunks=chunks_text, query=query)
