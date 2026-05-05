"""
Transliterate more urdu-instruct examples to Roman Urdu (batch 2).

Script 05 already transliterated 5,000 examples (seed=42).
This continues with a different seed to get fresh examples, boosting
Roman Urdu from ~9% to ~20% of training data.

Same approach as 05 — GPT-4o-mini, batched, resumable, budget-capped.

Run with:
    python scripts/13_transliterate_more_roman.py --n 5000 --budget 1.50

Resume:
    python scripts/13_transliterate_more_roman.py --n 5000 --budget 1.50 --resume
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

load_dotenv()

BATCH_SIZE = 5
MODEL = "gpt-4o-mini"
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60
SLEEP_BETWEEN = 0.5
OUTPUT_DIR = Path("data/roman_urdu_v2")
OUTPUT_FILE = OUTPUT_DIR / "transliterated_v2.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

SYSTEM_PROMPT = (
    "Convert Urdu script to Roman Urdu as a Pakistani would type on WhatsApp. "
    "Natural spellings (bohat, kya, hai). Keep English words in English. "
    "Do NOT translate. Only change script. Return same JSON structure."
)


def build_user_message(examples):
    items = []
    for i, ex in enumerate(examples):
        items.append({
            "id": i,
            "instruction": ex["instruction"],
            "output": ex["output"],
        })
    return json.dumps(items, ensure_ascii=False)


def parse_response(response_text):
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"done": 0, "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Number of examples")
    parser.add_argument("--budget", type=float, default=1.50, help="Max spend in dollars")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (different from script 05's seed=42)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    print(f"Loading urdu-instruct dataset...")
    ds = load_dataset("large-traversaal/urdu-instruct", split="train")
    print(f"Loaded {len(ds):,} examples")

    # Exclude indices already used by script 05 (seed=42, n=5000)
    random.seed(42)
    used_indices = set(random.sample(range(len(ds)), min(5000, len(ds))))
    print(f"Excluding {len(used_indices):,} indices already transliterated by script 05")

    available = [i for i in range(len(ds)) if i not in used_indices]
    random.seed(args.seed)
    indices = random.sample(available, min(args.n, len(available)))

    if args.resume:
        progress = load_progress()
        start_from = progress["done"]
        print(f"Resuming from example {start_from} (${progress['total_cost']:.4f} spent)")
    else:
        progress = {"done": 0, "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}
        start_from = 0
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")

    total_to_process = len(indices) - start_from
    batches = (total_to_process + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nPlan:")
    print(f"  Fresh examples to transliterate: {total_to_process:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  API calls needed: {batches:,}")
    print(f"  Budget: ${args.budget:.2f}")
    print(f"\nStarting...\n")

    failed_batches = 0

    for batch_idx in range(batches):
        batch_start = start_from + batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(indices))
        batch_indices = indices[batch_start:batch_end]
        batch_examples = [ds[i] for i in batch_indices]

        user_msg = build_user_message(batch_examples)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
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

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        batch_cost = (input_tokens * INPUT_COST_PER_M + output_tokens * OUTPUT_COST_PER_M) / 1_000_000

        progress["total_input_tokens"] += input_tokens
        progress["total_output_tokens"] += output_tokens
        progress["total_cost"] += batch_cost

        if progress["total_cost"] >= args.budget:
            print(f"\n  BUDGET LIMIT: ${progress['total_cost']:.4f} >= ${args.budget:.2f}")
            break

        parsed = parse_response(response.choices[0].message.content)

        if parsed is None:
            print(f"  Batch {batch_idx + 1}: parse failed. Skipping.")
            failed_batches += 1
            continue

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
                "source": "urdu-instruct-roman-v2",
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

        progress["done"] = batch_start + len(parsed)
        save_progress(progress)

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            pct = 100 * progress["done"] / total_to_process
            print(
                f"  Batch {batch_idx + 1}/{batches} | "
                f"{progress['done']:,} done ({pct:.1f}%) | "
                f"Cost: ${progress['total_cost']:.4f}"
            )

        time.sleep(SLEEP_BETWEEN)

    out_f.close()

    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"{'='*50}")
    print(f"Examples transliterated: {progress['done']:,}")
    print(f"Failed batches: {failed_batches}")
    print(f"Total cost: ${progress['total_cost']:.4f}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
