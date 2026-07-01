"""Build a rank-bm25 sidecar over the Qdrant Urdu Wikipedia collection.

Scrolls all chunks from /vol/qdrant_data_v1, tokenizes with a simple
Urdu-aware splitter (strip Urdu + ASCII punctuation, split on whitespace),
constructs BM25Okapi, and pickles to /vol/bm25_index_v1.pkl. Saves a parallel
chunk_ids JSON so we can map BM25 hits back to Qdrant point IDs.

Runs on Modal CPU container (no GPU needed; ~10 min, ~$0.20).

    modal run scripts/20c_build_bm25.py
    modal run scripts/20c_build_bm25.py --limit 1000     # smoke
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "qdrant-client",
        "rank-bm25>=0.2.2",
        "numpy",
    )
)

app = modal.App("urdu-rag-bm25")
vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=False)

QDRANT_VOL_PATH = "/vol/qdrant_data_v1"
COLLECTION = "urdu_wikipedia_v1"
BM25_OUT = "/vol/bm25_index_v1.pkl"
IDS_OUT = "/vol/bm25_chunk_ids_v1.json"

URDU_PUNCT = "۔،؟!؛٪٫٬"
ASCII_PUNCT = ".,!?;:()[]{}\"'-·–—"
_TRANSLATE_TABLE = str.maketrans({c: " " for c in URDU_PUNCT + ASCII_PUNCT})


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in text.translate(_TRANSLATE_TABLE).split() if t]


@app.function(
    cpu=4.0,
    memory=16384,
    image=image,
    volumes={"/vol": vol},
    timeout=3600,
)
def build_bm25(limit: int | None = None, scroll_batch: int = 2000):
    import json
    import pickle
    import time
    from pathlib import Path
    from qdrant_client import QdrantClient
    from rank_bm25 import BM25Okapi

    vol.reload()
    qdrant = QdrantClient(path=QDRANT_VOL_PATH)
    info = qdrant.get_collection(COLLECTION)
    n_total = info.points_count
    target = min(limit, n_total) if limit else n_total
    print(f"Collection {COLLECTION}: {n_total:,} points. Target: {target:,}")

    tokenized: list[list[str]] = []
    chunk_ids: list[str] = []
    titles: list[str] = []

    offset = None
    seen = 0
    t0 = time.time()
    while True:
        pts, offset = qdrant.scroll(
            collection_name=COLLECTION,
            limit=scroll_batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break
        for pt in pts:
            payload = pt.payload or {}
            content = payload.get("content") or ""
            meta = payload.get("meta") or {}
            tokens = tokenize(content)
            if not tokens:
                continue
            tokenized.append(tokens)
            chunk_ids.append(str(pt.id))
            titles.append(meta.get("title") or payload.get("title") or "")
            seen += 1
            if limit and seen >= limit:
                break
        if seen >= target:
            break
        if offset is None:
            break
        if seen % 20000 < scroll_batch:
            rate = seen / max(time.time() - t0, 0.001)
            print(f"  scrolled {seen:,}/{target:,}  ({rate:.0f} pts/s)")

    print(f"Tokenized {seen:,} chunks in {time.time() - t0:.0f}s. Avg tokens: {sum(len(t) for t in tokenized) / max(len(tokenized), 1):.1f}")

    print("Building BM25Okapi (this is the slow part)...")
    t1 = time.time()
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 built in {time.time() - t1:.0f}s. corpus_size={bm25.corpus_size}")

    Path(BM25_OUT).parent.mkdir(parents=True, exist_ok=True)
    print(f"Pickling to {BM25_OUT}...")
    with open(BM25_OUT, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids, "titles": titles}, f, protocol=pickle.HIGHEST_PROTOCOL)
    Path(IDS_OUT).write_text(json.dumps(chunk_ids))
    vol.commit()
    size_mb = Path(BM25_OUT).stat().st_size / 1024**2
    print(f"DONE. {BM25_OUT} = {size_mb:.1f} MB")
    return {"chunks": seen, "size_mb": size_mb, "elapsed": time.time() - t0}


@app.local_entrypoint()
def main(limit: int = 0):
    result = build_bm25.remote(limit=limit or None)
    print(f"Job result: {result}")
