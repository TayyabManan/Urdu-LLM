"""Index Urdu Wikipedia into Qdrant — run on Modal H100 (fast, ~30 min, ~$2.25).

CPU path is too slow for bge-m3 (62 days for full corpus). This runs the same
chunking + embedding logic on Modal GPU, writes Qdrant data to the Modal volume,
then you download the directory to your local machine.

Deploy + run:
    modal run scripts/20a_index_wikipedia_modal.py                    # full 200k
    modal run scripts/20a_index_wikipedia_modal.py --limit 100        # smoke

Download to local after job completes:
    modal volume get urdu-llm-vol qdrant_data_v1 ./vector_db/
    # creates ./vector_db/qdrant_data_v1/  — rename to qdrant_local:
    mv ./vector_db/qdrant_data_v1 ./vector_db/qdrant_local

Then build_rag_pipeline() picks it up automatically.
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "haystack-ai>=2.29.0",
        "qdrant-haystack",
        "qdrant-client",
        "sentence-transformers>=3.0",
        "datasets>=2.19.0",
        "torch",
    )
    .add_local_python_source("src", copy=True)
)

app = modal.App("urdu-rag-index")
vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=False)

EMBEDDER_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
QDRANT_VOL_PATH = "/vol/qdrant_data_v1"
COLLECTION = "urdu_wikipedia_v1"
PROGRESS_VOL_PATH = "/vol/qdrant_data_v1_progress.json"


@app.function(
    gpu="H100",
    image=image,
    volumes={"/vol": vol},
    timeout=7200,
)
def index_full(limit: int | None = None, chunk_size: int = 512, overlap: int = 64,
               batch_size: int = 64, resume: bool = True):
    import json
    import time
    from pathlib import Path
    from datasets import load_dataset
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.utils import ComponentDevice
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    from src.rag.chunking import chunk_text

    vol.reload()
    Path(QDRANT_VOL_PATH).mkdir(parents=True, exist_ok=True)

    print(f"Loading wikimedia/wikipedia 20231101.ur ...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ur", split="train")
    print(f"Loaded {len(ds):,} articles")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"Limit applied: {len(ds):,} articles")

    progress_path = Path(PROGRESS_VOL_PATH)
    if resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        done_ids = set(progress.get("done_article_ids", []))
        total_chunks = progress.get("total_chunks", 0)
        print(f"Resume: {len(done_ids):,} articles, {total_chunks:,} chunks already indexed")
    else:
        done_ids = set()
        total_chunks = 0

    store = QdrantDocumentStore(
        path=QDRANT_VOL_PATH,
        index=COLLECTION,
        embedding_dim=EMBED_DIM,
        recreate_index=False,
        similarity="cosine",
        return_embedding=False,
    )

    print(f"Loading embedder {EMBEDDER_MODEL} on GPU...")
    embedder = SentenceTransformersDocumentEmbedder(
        model=EMBEDDER_MODEL,
        device=ComponentDevice.from_str("cuda"),
        batch_size=batch_size,
        progress_bar=False,
    )
    embedder.warm_up()
    print("Embedder ready.")

    flush_every = 1024
    docs_buffer: list[Document] = []
    articles_this_run = 0
    chunks_this_run = 0
    t0 = time.time()
    last_commit = t0

    def flush(buffer):
        if not buffer:
            return 0
        embedded = embedder.run(documents=buffer)["documents"]
        store.write_documents(embedded)
        return len(embedded)

    def save_progress():
        progress_path.write_text(json.dumps({
            "done_article_ids": sorted(done_ids),
            "total_chunks": total_chunks,
        }))
        vol.commit()

    for art in ds:
        aid = art["id"]
        if aid in done_ids:
            continue
        text = art["text"]
        if not text.strip():
            done_ids.add(aid)
            continue
        title = art["title"]
        for ci, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
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
        articles_this_run += 1

        if len(docs_buffer) >= flush_every:
            wrote = flush(docs_buffer)
            chunks_this_run += wrote
            total_chunks += wrote
            docs_buffer = []
            if time.time() - last_commit > 60:
                save_progress()
                last_commit = time.time()
            rate_art = articles_this_run / max(time.time() - t0, 0.001)
            rate_chunk = chunks_this_run / max(time.time() - t0, 0.001)
            print(f"  {articles_this_run:,} new art  {chunks_this_run:,} new chunks  "
                  f"({rate_art:.1f} art/s, {rate_chunk:.0f} chunks/s)")

    if docs_buffer:
        wrote = flush(docs_buffer)
        chunks_this_run += wrote
        total_chunks += wrote

    save_progress()
    elapsed = time.time() - t0
    print(f"\nDONE: {articles_this_run:,} new articles  {chunks_this_run:,} new chunks  {elapsed:.0f}s")
    print(f"Total chunks in store: {total_chunks:,}")
    print(f"Qdrant data: {QDRANT_VOL_PATH}")
    print(f"\nTo download to local:")
    print(f"  modal volume get urdu-llm-vol qdrant_data_v1 ./vector_db/")
    print(f"  mv ./vector_db/qdrant_data_v1 ./vector_db/qdrant_local")
    return {"articles": articles_this_run, "chunks": chunks_this_run, "elapsed": elapsed}


@app.local_entrypoint()
def main(limit: int = 0, batch_size: int = 64, resume: bool = True):
    """CLI: modal run scripts/20a_index_wikipedia_modal.py --limit 100"""
    result = index_full.remote(
        limit=limit or None,
        batch_size=batch_size,
        resume=resume,
    )
    print(f"Job result: {result}")
