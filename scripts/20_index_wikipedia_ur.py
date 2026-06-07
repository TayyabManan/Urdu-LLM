"""Index Urdu Wikipedia (wikimedia/wikipedia 20231101.ur) into local Qdrant.

Pipeline: HF dataset → chunk on Urdu sentence boundaries → embed via BAAI/bge-m3
         → write to QdrantDocumentStore at ./vector_db/qdrant_local/.

Usage:
    # Smoke (100 articles, ~1 min on CPU)
    python scripts/20_index_wikipedia_ur.py --limit 100

    # Full (~200k articles)
    python scripts/20_index_wikipedia_ur.py --device cuda      # if GPU available
    python scripts/20_index_wikipedia_ur.py                    # CPU, slow

Resume: re-running with same args appends only new docs (dedup by article id).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.chunking import chunk_text  # noqa: E402

DEFAULT_QDRANT_PATH = ROOT / "vector_db" / "qdrant_local"
DEFAULT_COLLECTION = "urdu_wikipedia_v1"
DEFAULT_PROGRESS = ROOT / "vector_db" / "index_progress.json"
EMBEDDER_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {"done_article_ids": [], "total_chunks": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress), encoding="utf-8")


def build_store(qdrant_path: Path, collection: str):
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    return QdrantDocumentStore(
        path=str(qdrant_path),
        index=collection,
        embedding_dim=EMBED_DIM,
        recreate_index=False,
        similarity="cosine",
        return_embedding=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N articles (smoke testing)")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Embedder batch size")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--resume", action="store_true",
                        help="Skip article ids already in progress.json")
    args = parser.parse_args()

    from datasets import load_dataset
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.utils import ComponentDevice

    print(f"Loading wikimedia/wikipedia 20231101.ur...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ur", split="train")
    print(f"Loaded {len(ds):,} articles")

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"Limit applied: {len(ds):,} articles")

    progress = load_progress(DEFAULT_PROGRESS) if args.resume else {"done_article_ids": [], "total_chunks": 0}
    done_ids = set(progress["done_article_ids"])
    if done_ids:
        print(f"Resume: {len(done_ids):,} articles already indexed, {progress['total_chunks']:,} chunks")

    args.qdrant_path.mkdir(parents=True, exist_ok=True)
    store = build_store(args.qdrant_path, args.collection)

    print(f"Building embedder ({EMBEDDER_MODEL} on {args.device})...")
    embedder = SentenceTransformersDocumentEmbedder(
        model=EMBEDDER_MODEL,
        device=ComponentDevice.from_str(args.device) if args.device else None,
        batch_size=args.batch_size,
        progress_bar=True,
    )
    embedder.warm_up()

    docs_buffer: list[Document] = []
    flush_every = 256  # docs per Qdrant write batch
    articles_done_this_run = 0
    chunks_done_this_run = 0
    t0 = time.time()

    def flush(buffer: list[Document]) -> int:
        if not buffer:
            return 0
        embedded = embedder.run(documents=buffer)["documents"]
        store.write_documents(embedded)
        return len(embedded)

    for art in ds:
        aid = art["id"]
        if aid in done_ids:
            continue
        title = art["title"]
        text = art["text"]
        if not text.strip():
            done_ids.add(aid)
            continue

        for ci, chunk in enumerate(chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)):
            docs_buffer.append(Document(
                content=chunk.text,
                meta={
                    "article_id": aid,
                    "title": title,
                    "url": art.get("url"),
                    "chunk_idx": ci,
                    "approx_tokens": chunk.approx_tokens,
                },
            ))

        done_ids.add(aid)
        articles_done_this_run += 1

        if len(docs_buffer) >= flush_every:
            wrote = flush(docs_buffer)
            chunks_done_this_run += wrote
            progress["total_chunks"] += wrote
            docs_buffer = []
            progress["done_article_ids"] = sorted(done_ids)
            save_progress(DEFAULT_PROGRESS, progress)
            rate = articles_done_this_run / max(time.time() - t0, 0.001)
            print(f"  {articles_done_this_run:,} articles  {chunks_done_this_run:,} chunks  ({rate:.1f} art/s)")

    if docs_buffer:
        wrote = flush(docs_buffer)
        chunks_done_this_run += wrote
        progress["total_chunks"] += wrote

    progress["done_article_ids"] = sorted(done_ids)
    save_progress(DEFAULT_PROGRESS, progress)

    elapsed = time.time() - t0
    print(f"\nDONE: {articles_done_this_run:,} new articles  {chunks_done_this_run:,} new chunks  {elapsed:.0f}s")
    print(f"Total chunks in store: {progress['total_chunks']:,}")
    print(f"Qdrant path: {args.qdrant_path}")
    print(f"Collection: {args.collection}")


if __name__ == "__main__":
    main()
