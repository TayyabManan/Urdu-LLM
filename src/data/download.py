"""
Download all datasets, map to Alpaca format, merge into one unified JSONL.

Datasets:
    1. large-traversaal/urdu-instruct    (~51k, Urdu script)
    2. CohereLabs/aya_dataset            (~700 Urdu subset)
    3. Redgerd/roman-urdu-alpaca-qa-mix  (~1k, Roman Urdu)
    4. Local transliterated Roman Urdu   (~5k, from GPT-4o-mini)

Output:
    data/combined/train.jsonl   — all examples in Alpaca format
    data/combined/stats.json    — counts per source

Run with:
    python -m src.data.download
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/combined")
OUTPUT_FILE = OUTPUT_DIR / "train.jsonl"
STATS_FILE = OUTPUT_DIR / "stats.json"
ROMAN_URDU_FILE = Path("data/roman_urdu/transliterated.jsonl")


def to_alpaca(instruction: str, input_: str, output: str, source: str) -> dict:
    """Standardize to Alpaca format with source tag."""
    return {
        "instruction": instruction.strip(),
        "input": input_.strip(),
        "output": output.strip(),
        "source": source,
    }


def load_urdu_instruct() -> list[dict]:
    """
    large-traversaal/urdu-instruct — ~51k GPT-4o generated Urdu instruction pairs.
    This is our primary dataset.
    """
    log.info("Loading urdu-instruct...")
    ds = load_dataset("large-traversaal/urdu-instruct", split="train")

    log.info(f"  Columns: {ds.column_names}")
    log.info(f"  Size: {len(ds):,}")

    examples = []
    for ex in ds:
        # Map columns — urdu-instruct uses 'instruction' and 'output'
        instruction = ex.get("instruction", ex.get("prompt", ""))
        output = ex.get("output", ex.get("response", ""))

        if not instruction or not output:
            continue

        examples.append(to_alpaca(
            instruction=instruction,
            input_="",
            output=output,
            source="urdu-instruct",
        ))

    log.info(f"  Kept: {len(examples):,}")
    return examples


def load_aya_urdu() -> list[dict]:
    """
    CohereLabs/aya_dataset — human-written multilingual instruction data.
    We filter to Urdu only (~700 examples).
    """
    log.info("Loading Aya dataset (filtering to Urdu)...")
    ds = load_dataset("CohereForAI/aya_dataset", split="train")

    log.info(f"  Columns: {ds.column_names}")
    log.info(f"  Total (all languages): {len(ds):,}")

    # Find the language column
    lang_col = None
    for candidate in ["language", "language_code", "lang"]:
        if candidate in ds.column_names:
            lang_col = candidate
            break

    if lang_col is None:
        log.warning("  Could not find language column! Skipping Aya.")
        return []

    # Filter to Urdu
    ds_urdu = ds.filter(
        lambda ex: "urdu" in ex[lang_col].lower() or ex[lang_col].lower() == "urd"
    )
    log.info(f"  Urdu examples: {len(ds_urdu):,}")

    # Map columns — Aya typically uses 'inputs' and 'targets'
    # But column names vary, so detect them
    examples = []
    for ex in ds_urdu:
        instruction = ex.get("inputs", ex.get("instruction", ex.get("prompt", "")))
        output = ex.get("targets", ex.get("output", ex.get("response", "")))

        if not instruction or not output:
            continue

        examples.append(to_alpaca(
            instruction=instruction,
            input_="",
            output=output,
            source="aya-urdu",
        ))

    log.info(f"  Kept: {len(examples):,}")
    return examples


def load_redgerd_roman() -> list[dict]:
    """
    Redgerd/roman-urdu-alpaca-qa-mix — ~1k Roman Urdu instruction pairs.
    Small but ready-to-use Roman Urdu data.
    """
    log.info("Loading Redgerd Roman Urdu dataset...")

    try:
        ds = load_dataset("Redgerd/roman-urdu-alpaca-qa-mix", split="train")
    except Exception as e:
        log.warning(f"  Failed to load Redgerd dataset: {e}")
        log.warning("  Skipping. This is non-critical.")
        return []

    log.info(f"  Columns: {ds.column_names}")
    log.info(f"  Size: {len(ds):,}")

    examples = []
    for ex in ds:
        instruction = ex.get("instruction", ex.get("prompt", ""))
        output = ex.get("output", ex.get("response", ""))
        input_ = ex.get("input", "")

        if not instruction or not output:
            continue

        examples.append(to_alpaca(
            instruction=instruction,
            input_=input_ or "",
            output=output,
            source="redgerd-roman",
        ))

    log.info(f"  Kept: {len(examples):,}")
    return examples


def load_transliterated_roman() -> list[dict]:
    """
    Our GPT-4o-mini transliterated Roman Urdu data from urdu-instruct.
    Already in Alpaca format.
    """
    log.info(f"Loading transliterated Roman Urdu from {ROMAN_URDU_FILE}...")

    if not ROMAN_URDU_FILE.exists():
        log.warning(f"  File not found: {ROMAN_URDU_FILE}")
        log.warning("  Run scripts/05_transliterate_to_roman.py first.")
        return []

    examples = []
    with open(ROMAN_URDU_FILE, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            # Already has instruction/input/output/source
            examples.append(to_alpaca(
                instruction=ex.get("instruction", ""),
                input_=ex.get("input", ""),
                output=ex.get("output", ""),
                source="urdu-instruct-roman",
            ))

    log.info(f"  Loaded: {len(examples):,}")
    return examples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 50)
    log.info("DOWNLOADING AND MERGING ALL DATASETS")
    log.info("=" * 50)

    # Load all sources
    all_examples = []
    stats = {}

    sources = [
        ("urdu-instruct", load_urdu_instruct),
        ("aya-urdu", load_aya_urdu),
        ("redgerd-roman", load_redgerd_roman),
        ("urdu-instruct-roman", load_transliterated_roman),
    ]

    for name, loader in sources:
        log.info("")
        examples = loader()
        all_examples.extend(examples)
        stats[name] = len(examples)

    # Write combined JSONL
    log.info(f"\nWriting {len(all_examples):,} examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write stats
    stats["total"] = len(all_examples)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    log.info("")
    log.info("=" * 50)
    log.info("DATASET SUMMARY")
    log.info("=" * 50)
    for name, count in stats.items():
        if name == "total":
            continue
        pct = 100 * count / stats["total"] if stats["total"] > 0 else 0
        bar = "█" * int(pct / 2)
        log.info(f"  {name:25s} {count:>7,} ({pct:5.1f}%) {bar}")
    log.info(f"  {'─' * 45}")
    log.info(f"  {'TOTAL':25s} {stats['total']:>7,}")
    log.info(f"\nOutput: {OUTPUT_FILE}")
    log.info(f"Stats:  {STATS_FILE}")


if __name__ == "__main__":
    main()
