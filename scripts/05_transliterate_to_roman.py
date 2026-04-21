"""
Transliterate urdu-instruct examples from Urdu script → Roman Urdu using GPT-4o-mini.

Token-saving strategies:
    1. Batch 5 examples per API call (system prompt sent once, not 5x)
    2. Minimal system prompt (39 words vs 100+)
    3. Compact input format (no verbose labels)
    4. Saves progress after each batch — resume if interrupted
    5. Tracks token usage and cost in real-time
    6. Stops automatically if cost exceeds budget

Run with:
    python scripts/05_transliterate_to_roman.py --n 5000 --budget 2.50

Resume interrupted run (skips already-done examples):
    python scripts/05_transliterate_to_roman.py --n 5000 --budget 2.50 --resume
"""

import argparse
import json
import os
import time
import random
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI

# Load .env file from project root
load_dotenv()

# ─── Config ───

BATCH_SIZE = 5            # Examples per API call. Higher = fewer calls = less prompt overhead.
MODEL = "gpt-4o-mini"     # Cheapest model that handles Urdu well.
INPUT_COST_PER_M = 0.15   # $/1M input tokens
OUTPUT_COST_PER_M = 0.60  # $/1M output tokens
SLEEP_BETWEEN = 0.5       # Seconds between API calls. Avoids rate limits.
OUTPUT_DIR = Path("data/roman_urdu")
OUTPUT_FILE = OUTPUT_DIR / "transliterated.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Minimal system prompt — every token here is repeated per API call.
# Shorter = cheaper. This is 39 words vs a typical 100+ word prompt.
SYSTEM_PROMPT = (
    "Convert Urdu script to Roman Urdu as a Pakistani would type on WhatsApp. "
    "Natural spellings (bohat, kya, hai). Keep English words in English. "
    "Do NOT translate. Only change script. Return same JSON structure."
)


def build_user_message(examples):
    """
    Pack multiple examples into one message.

    Format is compact — no verbose labels, just numbered JSON blocks.
    The model returns the same structure with Roman Urdu text.
    """
    items = []
    for i, ex in enumerate(examples):
        items.append({
            "id": i,
            "instruction": ex["instruction"],
            "output": ex["output"],
        })
    return json.dumps(items, ensure_ascii=False)


def parse_response(response_text, batch_size):
    """Parse the model's JSON response. Handle common failure modes."""
    # Try parsing as JSON array
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Sometimes model wraps in markdown code block
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        # Remove ```json and ``` markers
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return None  # Failed to parse


def load_progress():
    """Load progress from last run. Returns number of examples already done."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"done": 0, "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Number of examples to transliterate")
    parser.add_argument("--budget", type=float, default=2.50, help="Max spend in dollars")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()  # Uses OPENAI_API_KEY env var

    # Load dataset
    print(f"Loading urdu-instruct dataset...")
    ds = load_dataset("large-traversaal/urdu-instruct", split="train")
    print(f"Loaded {len(ds):,} examples")

    # Sample N examples randomly (deterministic with seed)
    random.seed(args.seed)
    indices = random.sample(range(len(ds)), min(args.n, len(ds)))

    # Resume handling
    if args.resume:
        progress = load_progress()
        start_from = progress["done"]
        print(f"Resuming from example {start_from} (${progress['total_cost']:.4f} spent so far)")
    else:
        progress = {"done": 0, "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}
        start_from = 0
        # Clear output file if starting fresh
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

    # Open output file in append mode
    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")

    total_to_process = len(indices) - start_from
    batches = (total_to_process + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nPlan:")
    print(f"  Examples to process: {total_to_process:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  API calls needed: {batches:,}")
    print(f"  Budget: ${args.budget:.2f}")
    print(f"  Model: {MODEL}")
    print(f"\nStarting...\n")

    failed_batches = 0

    for batch_idx in range(batches):
        batch_start = start_from + batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(indices))
        batch_indices = indices[batch_start:batch_end]
        batch_examples = [ds[i] for i in batch_indices]

        # Build API request
        user_msg = build_user_message(batch_examples)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,  # Low temp = more consistent transliteration
                max_tokens=4096,
            )
        except Exception as e:
            print(f"  API error on batch {batch_idx + 1}: {e}")
            failed_batches += 1
            if failed_batches > 10:
                print("Too many failures. Stopping.")
                break
            time.sleep(2)
            continue

        # Track tokens and cost
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        batch_cost = (input_tokens * INPUT_COST_PER_M + output_tokens * OUTPUT_COST_PER_M) / 1_000_000

        progress["total_input_tokens"] += input_tokens
        progress["total_output_tokens"] += output_tokens
        progress["total_cost"] += batch_cost

        # Budget check — stop before going over
        if progress["total_cost"] >= args.budget:
            print(f"\n  BUDGET LIMIT REACHED: ${progress['total_cost']:.4f} >= ${args.budget:.2f}")
            print(f"  Stopping. {progress['done']} examples completed.")
            break

        # Parse response
        parsed = parse_response(response.choices[0].message.content, len(batch_examples))

        if parsed is None:
            print(f"  Batch {batch_idx + 1}: failed to parse response. Skipping.")
            failed_batches += 1
            continue

        # Write results
        for j, item in enumerate(parsed):
            if j >= len(batch_examples):
                break

            original = batch_examples[j]
            result = {
                "instruction": item.get("instruction", ""),
                "input": "",
                "output": item.get("output", ""),
                "original_instruction": original["instruction"],
                "original_output": original["output"],
                "source": "urdu-instruct-roman",
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

        progress["done"] = batch_start + len(parsed)
        save_progress(progress)

        # Progress update every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            pct = 100 * progress["done"] / total_to_process
            print(
                f"  Batch {batch_idx + 1}/{batches} | "
                f"{progress['done']:,} done ({pct:.1f}%) | "
                f"Cost: ${progress['total_cost']:.4f} | "
                f"Tokens: {progress['total_input_tokens'] + progress['total_output_tokens']:,}"
            )

        time.sleep(SLEEP_BETWEEN)

    out_f.close()

    # Final summary
    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"{'='*50}")
    print(f"Examples transliterated: {progress['done']:,}")
    print(f"Failed batches: {failed_batches}")
    print(f"Total tokens: {progress['total_input_tokens'] + progress['total_output_tokens']:,}")
    print(f"  Input:  {progress['total_input_tokens']:,}")
    print(f"  Output: {progress['total_output_tokens']:,}")
    print(f"Total cost: ${progress['total_cost']:.4f}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"\nTo resume if interrupted: python scripts/05_transliterate_to_roman.py --n {args.n} --budget {args.budget} --resume")


if __name__ == "__main__":
    main()
