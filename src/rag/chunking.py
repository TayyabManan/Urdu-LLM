"""Urdu-aware sentence chunker for RAG indexing.

Splits text on Urdu sentence boundaries (۔ ! ؟) and groups sentences into chunks
targeting ~chunk_size tokens with overlap. Token counting uses the embedder's
own tokenizer when available; falls back to a 1 char ≈ 0.4 token heuristic for
Urdu script (Urdu BPE tokens average ~2.5 chars).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

URDU_SENTENCE_END = re.compile(r"(?<=[۔!؟\.])\s+")


@dataclass
class Chunk:
    text: str
    char_len: int
    approx_tokens: int


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = URDU_SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def approx_token_count(text: str) -> int:
    """Cheap heuristic without loading a tokenizer. Urdu BPE ≈ 2.5 chars/token."""
    return max(1, int(len(text) / 2.5))


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    token_fn=approx_token_count,
) -> list[Chunk]:
    """Group sentences into ~chunk_size-token chunks with ~overlap-token overlap.

    Greedy pack-sentences-until-full; overlap by re-using the trailing sentences
    of the previous chunk whose total tokens >= overlap.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = token_fn(sent)

        # Edge case: single sentence longer than chunk_size — emit alone.
        if sent_tokens >= chunk_size and not current:
            chunks.append(Chunk(text=sent, char_len=len(sent), approx_tokens=sent_tokens))
            continue

        if current_tokens + sent_tokens > chunk_size and current:
            chunk_text_str = " ".join(current)
            chunks.append(
                Chunk(text=chunk_text_str, char_len=len(chunk_text_str), approx_tokens=current_tokens)
            )
            # Start next chunk with trailing-sentence overlap
            overlap_sents: list[str] = []
            overlap_tokens = 0
            for s in reversed(current):
                t = token_fn(s)
                if overlap_tokens + t > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tokens += t
            current = overlap_sents
            current_tokens = overlap_tokens

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunk_text_str = " ".join(current)
        chunks.append(
            Chunk(text=chunk_text_str, char_len=len(chunk_text_str), approx_tokens=current_tokens)
        )

    return chunks
