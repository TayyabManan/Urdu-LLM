"""
Combine original training data with new Roman Urdu + code data, reformat, upload.

Takes:
    - data/formatted/train.jsonl          (original 56,666 examples)
    - data/code_roman/code_roman.jsonl    (new code explanations in Roman Urdu)
    - data/roman_urdu_v2/transliterated_v2.jsonl  (new Roman Urdu transliterations)

Cleans new data, formats with Qwen chat template, combines with original,
shuffles, and uploads to Modal Volume.

Run with:
    python scripts/14_combine_and_upload_v2.py          # local combine + format
    modal run scripts/14_combine_and_upload_v2.py       # upload to Modal
"""

import json
import logging
import re
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Paths
ORIGINAL_FILE = Path("data/formatted/train.jsonl")
CODE_FILE = Path("data/code_roman/code_roman.jsonl")
ROMAN_V2_FILE = Path("data/roman_urdu_v2/transliterated_v2.jsonl")
OUTPUT_DIR = Path("data/formatted_v2")
OUTPUT_FILE = OUTPUT_DIR / "train.jsonl"
STATS_FILE = OUTPUT_DIR / "stats.json"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048
MIN_CHARS = 20
MAX_CHARS = 5000


def clean_example(ex: dict) -> bool:
    """Basic cleaning. Returns True if example should be kept."""
    instruction = ex.get("instruction", "").strip()
    output = ex.get("output", "").strip()

    if not instruction or not output:
        return False

    combined = len(instruction) + len(output)
    if combined < MIN_CHARS or combined > MAX_CHARS:
        return False

    return True


def load_new_data(filepath: Path, source_name: str) -> list[dict]:
    """Load and clean new data from generation scripts."""
    if not filepath.exists():
        log.info(f"  {filepath} not found — skipping")
        return []

    examples = []
    skipped = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            ex["source"] = source_name
            if clean_example(ex):
                examples.append(ex)
            else:
                skipped += 1

    log.info(f"  {filepath.name}: {len(examples):,} kept, {skipped} skipped")
    return examples


def format_example(ex: dict, tokenizer) -> dict | None:
    """Convert Alpaca-style example to Qwen chat template."""
    user_message = ex["instruction"]
    if ex.get("input", "").strip():
        user_message += f"\n\n{ex['input']}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ex["output"]},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )

    n_tokens = len(tokenizer.encode(text))

    if n_tokens > MAX_SEQ_LENGTH:
        return None

    return {
        "text": text,
        "source": ex.get("source", "unknown"),
        "n_tokens": n_tokens,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Load original formatted data ───
    log.info("Loading original formatted data...")
    original = []
    with open(ORIGINAL_FILE, encoding="utf-8") as f:
        for line in f:
            original.append(json.loads(line))
    log.info(f"  Original: {len(original):,} examples")

    # ─── Load new data ───
    log.info("\nLoading new data...")
    new_code = load_new_data(CODE_FILE, "code-roman-synthetic")
    new_roman = load_new_data(ROMAN_V2_FILE, "urdu-instruct-roman-v2")

    if not new_code and not new_roman:
        log.info("No new data found. Nothing to do.")
        return

    # ─── Format new data ───
    log.info(f"\nFormatting new data with Qwen chat template...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    new_formatted = []
    skipped_long = 0

    for ex in new_code + new_roman:
        result = format_example(ex, tokenizer)
        if result:
            new_formatted.append(result)
        else:
            skipped_long += 1

    log.info(f"  New formatted: {len(new_formatted):,}")
    log.info(f"  Skipped (>{MAX_SEQ_LENGTH} tokens): {skipped_long}")

    # ─── Combine and shuffle ───
    combined = original + new_formatted
    random.seed(42)
    random.shuffle(combined)

    log.info(f"\nCombined total: {len(combined):,}")

    # ─── Write ───
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ─── Stats ───
    source_counts = {}
    for ex in combined:
        src = ex.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    stats = {
        "original_count": len(original),
        "new_code_count": len([e for e in new_formatted if e["source"] == "code-roman-synthetic"]),
        "new_roman_count": len([e for e in new_formatted if e["source"] == "urdu-instruct-roman-v2"]),
        "total": len(combined),
        "skipped_long": skipped_long,
        "by_source": source_counts,
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"\n{'='*50}")
    log.info("COMBINE SUMMARY")
    log.info(f"{'='*50}")
    log.info(f"  Original:       {len(original):>7,}")
    log.info(f"  + Code Roman:   {stats['new_code_count']:>7,}")
    log.info(f"  + Roman v2:     {stats['new_roman_count']:>7,}")
    log.info(f"  = Total:        {len(combined):>7,}")

    log.info(f"\nBy source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(combined)
        log.info(f"  {source:30s} {count:>7,} ({pct:5.1f}%)")

    log.info(f"\nOutput: {OUTPUT_FILE}")
    log.info(f"Stats:  {STATS_FILE}")


# ─── Modal upload ───
try:
    import modal

    app = modal.App("upload-data-v2")
    vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=True)

    @app.function(volumes={"/vol": vol})
    def upload(data: bytes):
        import os
        os.makedirs("/vol/data", exist_ok=True)

        # Backup original
        if os.path.exists("/vol/data/train.jsonl"):
            os.rename("/vol/data/train.jsonl", "/vol/data/train_v1_backup.jsonl")
            print("Backed up original as train_v1_backup.jsonl")

        with open("/vol/data/train.jsonl", "wb") as f:
            f.write(data)

        lines = sum(1 for _ in open("/vol/data/train.jsonl", encoding="utf-8"))
        print(f"Written {lines:,} lines to /vol/data/train.jsonl")
        vol.commit()

    @app.local_entrypoint()
    def modal_main():
        # Run local combine first
        main()

        # Then upload
        print(f"\nUploading to Modal Volume...")
        with open(OUTPUT_FILE, "rb") as f:
            data = f.read()
        print(f"Size: {len(data) / 1024**2:.1f} MB")
        upload.remote(data)
        print("Done! New data on Modal Volume at /vol/data/train.jsonl")
        print("Original backed up as /vol/data/train_v1_backup.jsonl")

except ImportError:
    pass


if __name__ == "__main__":
    main()
