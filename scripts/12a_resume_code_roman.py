"""
Resume code Roman Urdu generation + combine both batches.

Run after 12_generate_code_roman.py finishes (or after resuming it):
    python scripts/12a_resume_code_roman.py

What it does:
    1. If code_roman.jsonl generation isn't done, resumes it
    2. Combines batch1 + batch2
    3. Shows stats
"""

import subprocess
import sys
from pathlib import Path

BATCH1 = Path("data/code_roman/code_roman_batch1.jsonl")
BATCH2 = Path("data/code_roman/code_roman.jsonl")
PROGRESS = Path("data/code_roman/progress.json")


def count_lines(filepath):
    if not filepath.exists():
        return 0
    return sum(1 for _ in open(filepath, encoding="utf-8"))


def main():
    print("=" * 50)
    print("Code Roman Urdu — Resume & Combine")
    print("=" * 50)

    b1_count = count_lines(BATCH1)
    b2_count = count_lines(BATCH2)
    print(f"Batch 1: {b1_count:,} examples")
    print(f"Batch 2: {b2_count:,} examples (current run)")

    # Check if generation needs resuming
    if PROGRESS.exists():
        import json
        with open(PROGRESS) as f:
            progress = json.load(f)
        print(f"Progress: {progress['done']} done, ${progress['total_cost']:.4f} spent")

    # Resume generation
    print("\nResuming generation...")
    result = subprocess.run(
        [sys.executable, "scripts/12_generate_code_roman.py", "--n", "3000", "--seed", "99", "--budget", "1.50", "--resume"],
        cwd=Path(__file__).parent.parent,
    )

    # Combine batches
    b2_count = count_lines(BATCH2)
    print(f"\nAfter resume — Batch 2: {b2_count:,} examples")

    if b1_count > 0 and b2_count > 0:
        print("Combining batch 1 + batch 2...")
        combined = Path("data/code_roman/temp_combined.jsonl")
        with open(combined, "w", encoding="utf-8") as out:
            for src in [BATCH1, BATCH2]:
                with open(src, encoding="utf-8") as f:
                    for line in f:
                        out.write(line)

        # Replace main file
        combined.replace(BATCH2)
        total = count_lines(BATCH2)
        print(f"Combined: {total:,} total code Roman examples")
    else:
        total = b2_count
        print(f"Total: {total:,} code Roman examples")

    print(f"\nNext step:")
    print(f"  modal run scripts/14_combine_and_upload_v2.py")
    print(f"  modal run scripts/15_retrain_v2.py")


if __name__ == "__main__":
    main()
