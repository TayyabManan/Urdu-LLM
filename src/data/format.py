"""
Format cleaned data into Qwen chat template for training.

Takes Alpaca-style JSONL and converts to the format the model actually trains on.
Each example becomes a full chat conversation with special tokens:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {instruction + input}<|im_end|>
    <|im_start|>assistant
    {output}<|im_end|>

Why this step is separate from cleaning:
    - Cleaning is model-agnostic (same cleaned data works for any model)
    - Formatting is model-specific (Qwen template ≠ Llama template ≠ Mistral template)
    - If you switch base models later, you re-run format.py, not clean.py

Input:  data/cleaned/train.jsonl
Output: data/formatted/train.jsonl     — ready for training
        data/formatted/stats.json      — token length stats

Run with:
    python -m src.data.format
"""

import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

INPUT_FILE = Path("data/cleaned/train.jsonl")
OUTPUT_DIR = Path("data/formatted")
OUTPUT_FILE = OUTPUT_DIR / "train.jsonl"
STATS_FILE = OUTPUT_DIR / "stats.json"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Using 7B tokenizer now — this is the real model
MAX_SEQ_LENGTH = 2048


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer — we need this to apply the chat template
    # and to measure token lengths accurately
    log.info(f"Loading tokenizer: {MODEL_NAME}")
    log.info("(Downloads ~1MB, not the full model)\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load cleaned data
    log.info(f"Loading cleaned data from {INPUT_FILE}...")
    examples = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    log.info(f"Loaded: {len(examples):,} examples\n")

    # Format and measure
    formatted = []
    token_lengths = []
    skipped_long = 0
    source_counts = {}

    for ex in examples:
        # Build user message: instruction + optional input
        user_message = ex["instruction"]
        if ex.get("input", "").strip():
            user_message += f"\n\n{ex['input']}"

        # Build conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ex["output"]},
        ]

        # Apply chat template — this adds <|im_start|>, <|im_end|> tokens
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Measure token length
        n_tokens = len(tokenizer.encode(text))

        # Skip if too long for our training window
        if n_tokens > MAX_SEQ_LENGTH:
            skipped_long += 1
            continue

        token_lengths.append(n_tokens)

        formatted.append({
            "text": text,
            "source": ex.get("source", "unknown"),
            "n_tokens": n_tokens,
        })

        source = ex.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    # Write formatted data
    log.info(f"Writing {len(formatted):,} formatted examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in formatted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    sorted_lengths = sorted(token_lengths)
    stats = {
        "input_total": len(examples),
        "output_total": len(formatted),
        "skipped_too_long": skipped_long,
        "token_lengths": {
            "min": sorted_lengths[0] if sorted_lengths else 0,
            "max": sorted_lengths[-1] if sorted_lengths else 0,
            "mean": sum(sorted_lengths) // len(sorted_lengths) if sorted_lengths else 0,
            "median": sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0,
            "p90": sorted_lengths[int(len(sorted_lengths) * 0.9)] if sorted_lengths else 0,
            "p95": sorted_lengths[int(len(sorted_lengths) * 0.95)] if sorted_lengths else 0,
            "p99": sorted_lengths[int(len(sorted_lengths) * 0.99)] if sorted_lengths else 0,
        },
        "by_source": source_counts,
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    log.info(f"\n{'='*50}")
    log.info("FORMAT SUMMARY")
    log.info(f"{'='*50}")
    log.info(f"  Input:          {len(examples):>7,}")
    log.info(f"  Formatted:      {len(formatted):>7,}")
    log.info(f"  Skipped (>{MAX_SEQ_LENGTH} tokens): {skipped_long:>7,}")

    log.info(f"\nToken length distribution:")
    log.info(f"  Min:    {stats['token_lengths']['min']:,}")
    log.info(f"  Median: {stats['token_lengths']['median']:,}")
    log.info(f"  Mean:   {stats['token_lengths']['mean']:,}")
    log.info(f"  P90:    {stats['token_lengths']['p90']:,}")
    log.info(f"  P95:    {stats['token_lengths']['p95']:,}")
    log.info(f"  P99:    {stats['token_lengths']['p99']:,}")
    log.info(f"  Max:    {stats['token_lengths']['max']:,}")

    log.info(f"\nBy source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(formatted)
        log.info(f"  {source:25s} {count:>7,} ({pct:5.1f}%)")

    log.info(f"\nOutput: {OUTPUT_FILE}")
    log.info(f"Stats:  {STATS_FILE}")
    log.info(f"\nThis file is ready for training.")


if __name__ == "__main__":
    main()
