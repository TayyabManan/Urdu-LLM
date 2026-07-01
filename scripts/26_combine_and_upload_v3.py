"""
Combine the v2 training set with the v3 synthetic data into a NEW train_v3.jsonl and
upload it to Modal — leaving the v2 train.jsonl on the volume untouched.

v3 adds three new ingredients on top of v2 (see scripts/23/24/25):
    - data/grammar_pairs/grammar_pairs.jsonl   (fix the grammar regression)
    - data/summ_reason/summ_reason.jsonl       (fix summarization + reasoning)
    - data/rag_triples/rag_triples.jsonl       (make the model RAG-aware)

Two things differ from the v2 combine (scripts/14):
  1. Per-example system prompt. RAG rows carry their own `system` (RAG_SYSTEM) and an
     `instruction` that is ALREADY the `{chunks}\n\nسوال:{query}` surface the /rag
     endpoint serves — reproducing that exact shape in training is the whole point of v3.
     Non-RAG rows have no `system`, so they default to "You are a helpful assistant."
     (same as v2).
  2. Dual token limit. RAG examples carry context and need room (<=3900); everything else
     keeps the v2 window (<=2048). scripts/27 trains at MAX_SEQ_LENGTH=4096, so a <=3900
     RAG row is never truncated.

The eval-contamination guard already ran at generation time (scripts/23/24/25), so this
script does NOT re-screen.

Run with:
    python scripts/26_combine_and_upload_v3.py     # local combine + format only (no $)
    modal run scripts/26_combine_and_upload_v3.py  # combine, then upload to Modal volume
"""

import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ─── Paths ───
V2_BASE_FILE = Path("data/formatted_v2/train.jsonl")       # already {text,source,n_tokens}
GRAMMAR_FILE = Path("data/grammar_pairs/grammar_pairs.jsonl")
SUMM_REASON_FILE = Path("data/summ_reason/summ_reason.jsonl")
RAG_FILE = Path("data/rag_triples/rag_triples.jsonl")

OUTPUT_DIR = Path("data/formatted_v3")
OUTPUT_FILE = OUTPUT_DIR / "train.jsonl"
STATS_FILE = OUTPUT_DIR / "stats.json"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS_NON_RAG = 2048    # matches v2's window
MAX_TOKENS_RAG = 3900        # context needs room; scripts/27 trains at 4096

DEFAULT_SYSTEM = "You are a helpful assistant."   # matches v2 (src/data/format.py:76)

# Non-RAG char sanity bounds (mirror scripts/14). RAG skips these — multi-chunk context
# legitimately exceeds 5000 chars; its 3900-token cap governs length instead.
MIN_CHARS = 20
MAX_CHARS = 5000


def alpaca_ok(ex: dict, char_bounds: bool) -> bool:
    """Keep rows with non-empty instruction+output; optional char bounds for non-RAG."""
    instruction = (ex.get("instruction") or "").strip()
    output = (ex.get("output") or "").strip()
    if not instruction or not output:
        return False
    if char_bounds:
        n = len(instruction) + len(output)
        if n < MIN_CHARS or n > MAX_CHARS:
            return False
    return True


def format_example(ex: dict, tokenizer, max_tokens: int) -> dict | None:
    """Alpaca/RAG row -> {text, source, n_tokens} via the Qwen chat template.

    System is per-example: RAG rows carry RAG_SYSTEM in `system`; everything else falls
    back to DEFAULT_SYSTEM. The RAG `instruction` is ALREADY the full
    {chunks}\\n\\nسوال:{query} surface — passed through verbatim (no rebuild)."""
    system = ex.get("system") or DEFAULT_SYSTEM
    user_message = ex["instruction"]
    if (ex.get("input") or "").strip():
        user_message += f"\n\n{ex['input']}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ex["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    n_tokens = len(tokenizer.encode(text))
    if n_tokens > max_tokens:
        return None
    return {
        "text": text,
        "source": ex.get("source", "unknown"),
        "n_tokens": n_tokens,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def format_new_set(path: Path, tokenizer, max_tokens: int, is_rag: bool,
                   counters: dict) -> list[dict]:
    """Load + format one new v3 file. Preserves each row's own `source`."""
    if not path.exists():
        log.info(f"  {path} MISSING — skipping")
        return []
    out = []
    skipped_clean = skipped_long = 0
    for ex in load_jsonl(path):
        if not alpaca_ok(ex, char_bounds=not is_rag):
            skipped_clean += 1
            continue
        rec = format_example(ex, tokenizer, max_tokens)
        if rec is None:
            skipped_long += 1
            continue
        out.append(rec)
    counters[path.name] = {"kept": len(out), "skipped_clean": skipped_clean,
                           "skipped_long": skipped_long, "limit": max_tokens}
    log.info(f"  {path.name}: {len(out):,} kept "
             f"(skipped {skipped_clean} clean / {skipped_long} >{max_tokens}tok)")
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── v2 base (already formatted — ingest as-is) ───
    log.info(f"Loading v2 base: {V2_BASE_FILE}")
    if not V2_BASE_FILE.exists():
        sys.exit(f"Missing {V2_BASE_FILE}. Build it first via scripts/14.")
    v2_base = load_jsonl(V2_BASE_FILE)
    log.info(f"  v2 base: {len(v2_base):,} examples (as-is)\n")

    # ─── New v3 sets (format with the Qwen template) ───
    log.info(f"Formatting v3 data with {MODEL_NAME} chat template...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    counters: dict = {}
    new_rows: list[dict] = []
    new_rows += format_new_set(GRAMMAR_FILE, tokenizer, MAX_TOKENS_NON_RAG, False, counters)
    new_rows += format_new_set(SUMM_REASON_FILE, tokenizer, MAX_TOKENS_NON_RAG, False, counters)
    new_rows += format_new_set(RAG_FILE, tokenizer, MAX_TOKENS_RAG, True, counters)

    if not new_rows:
        sys.exit("No new v3 rows formatted — aborting (check the data/ files).")

    # ─── Combine + shuffle (seed 42, same as v2) ───
    combined = v2_base + new_rows
    random.seed(42)
    random.shuffle(combined)
    log.info(f"\nCombined: {len(v2_base):,} v2 + {len(new_rows):,} v3 = {len(combined):,}")

    # ─── Write ───
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ─── Stats ───
    source_counts: dict = {}
    max_tok = 0
    for ex in combined:
        src = ex.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        max_tok = max(max_tok, ex.get("n_tokens", 0))

    stats = {
        "v2_base": len(v2_base),
        "v3_new": len(new_rows),
        "total": len(combined),
        "max_n_tokens": max_tok,
        "limits": {"non_rag": MAX_TOKENS_NON_RAG, "rag": MAX_TOKENS_RAG},
        "per_file": counters,
        "by_source": source_counts,
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'='*50}\nV3 COMBINE SUMMARY\n{'='*50}")
    log.info(f"  v2 base:    {len(v2_base):>7,}")
    log.info(f"  + v3 new:   {len(new_rows):>7,}")
    log.info(f"  = total:    {len(combined):>7,}")
    log.info(f"  max tokens: {max_tok:,} (rag<= {MAX_TOKENS_RAG}, non-rag<= {MAX_TOKENS_NON_RAG})")
    log.info(f"\nBy source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(combined)
        log.info(f"  {source:30s} {count:>7,} ({pct:5.1f}%)")
    log.info(f"\nOutput: {OUTPUT_FILE}\nStats:  {STATS_FILE}")
    log.info("\nReady for scripts/27. Upload with: modal run scripts/26_combine_and_upload_v3.py")


# ─── Modal upload (writes a NEW train_v3.jsonl; leaves the v2 train.jsonl untouched) ───
try:
    import modal

    app = modal.App("upload-data-v3")
    vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=True)

    @app.function(volumes={"/vol": vol})
    def upload(data: bytes):
        import os
        os.makedirs("/vol/data", exist_ok=True)
        # NEW file — do NOT touch /vol/data/train.jsonl (the v2 set stays intact).
        with open("/vol/data/train_v3.jsonl", "wb") as f:
            f.write(data)
        lines = sum(1 for _ in open("/vol/data/train_v3.jsonl", encoding="utf-8"))
        print(f"Written {lines:,} lines to /vol/data/train_v3.jsonl")
        vol.commit()

    @app.local_entrypoint()
    def modal_main():
        main()
        print("\nUploading to Modal volume...")
        with open(OUTPUT_FILE, "rb") as f:
            data = f.read()
        print(f"Size: {len(data) / 1024**2:.1f} MB")
        upload.remote(data)
        print("Done. v3 data at /vol/data/train_v3.jsonl (v2 /vol/data/train.jsonl untouched).")

except ImportError:
    pass


if __name__ == "__main__":
    main()
