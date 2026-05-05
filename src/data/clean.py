"""
Clean and filter the combined dataset.

Cleaning steps:
    1. Drop empty/whitespace-only instruction or output
    2. Drop exact duplicate instructions
    3. Drop too-short examples (<20 chars combined)
    4. Drop too-long examples (>5000 chars combined)
    5. Fix Roman Urdu entries with leftover Urdu script characters
    6. Flag examples with suspicious script mismatch
    7. Strip extra whitespace

Input:  data/combined/train.jsonl
Output: data/cleaned/train.jsonl         — cleaned dataset
        data/cleaned/removed.jsonl       — removed examples (for inspection)
        data/cleaned/stats.json          — cleaning stats

Run with:
    python -m src.data.clean
"""

import json
import logging
import re
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

INPUT_FILE = Path("data/combined/train.jsonl")
OUTPUT_DIR = Path("data/cleaned")
OUTPUT_FILE = OUTPUT_DIR / "train.jsonl"
REMOVED_FILE = OUTPUT_DIR / "removed.jsonl"
STATS_FILE = OUTPUT_DIR / "stats.json"

MIN_CHARS = 20       # Minimum combined length (instruction + output)
MAX_CHARS = 5000     # Maximum combined length
URDU_RANGE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')


def combined_length(ex: dict) -> int:
    return len(ex["instruction"]) + len(ex.get("input", "")) + len(ex["output"])


def urdu_char_count(text: str) -> int:
    return len(URDU_RANGE.findall(text))


def urdu_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Urdu/Arabic script."""
    urdu = urdu_char_count(text)
    alpha = len(re.findall(r'[a-zA-Z]', text)) + urdu
    if alpha == 0:
        return 0.0
    return urdu / alpha


def strip_urdu_from_roman(text: str) -> str:
    """
    Remove stray Urdu script characters from Roman Urdu text.
    Catches cases like 'ta تعطیلات' where transliteration was partial.
    Only removes isolated Urdu characters/words, not full Urdu text.
    """
    # If text is mostly Urdu (>50%), don't touch it — it's Urdu script data
    if urdu_ratio(text) > 0.5:
        return text
    # Remove Urdu characters from mostly-Roman text
    cleaned = URDU_RANGE.sub('', text)
    # Clean up double spaces left behind
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    return cleaned


def clean_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, strip edges."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    log.info("Loading data from %s...", INPUT_FILE)
    raw = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line))
    log.info(f"Loaded: {len(raw):,} examples\n")

    kept = []
    removed = []
    reasons = Counter()

    seen_instructions = set()

    for ex in raw:
        source = ex.get("source", "unknown")

        # ─── Step 1: Clean whitespace ───
        ex["instruction"] = clean_whitespace(ex["instruction"])
        ex["input"] = clean_whitespace(ex.get("input", ""))
        ex["output"] = clean_whitespace(ex["output"])

        # ─── Step 2: Drop empty fields ───
        if not ex["instruction"] or not ex["output"]:
            reasons["empty_field"] += 1
            removed.append({**ex, "removal_reason": "empty_field"})
            continue

        # ─── Step 3: Drop exact duplicate instructions ───
        instr_key = ex["instruction"].strip().lower()
        if instr_key in seen_instructions:
            reasons["duplicate"] += 1
            removed.append({**ex, "removal_reason": "duplicate"})
            continue
        seen_instructions.add(instr_key)

        # ─── Step 4: Drop too short ───
        length = combined_length(ex)
        if length < MIN_CHARS:
            reasons["too_short"] += 1
            removed.append({**ex, "removal_reason": "too_short"})
            continue

        # ─── Step 5: Drop too long ───
        if length > MAX_CHARS:
            reasons["too_long"] += 1
            removed.append({**ex, "removal_reason": "too_long"})
            continue

        # ─── Step 6: Fix stray Urdu in Roman Urdu entries ───
        if source in ("urdu-instruct-roman", "redgerd-roman"):
            ex["instruction"] = strip_urdu_from_roman(ex["instruction"])
            ex["output"] = strip_urdu_from_roman(ex["output"])

            # After stripping, check if anything meaningful is left
            if len(ex["instruction"].strip()) < 5 or len(ex["output"].strip()) < 5:
                reasons["roman_cleanup_empty"] += 1
                removed.append({**ex, "removal_reason": "roman_cleanup_empty"})
                continue

        # ─── Step 7: Script mismatch check ───
        # At least ONE of instruction or output should contain Urdu script.
        # This keeps translation tasks (Urdu instruction → English output)
        # and math problems (Urdu instruction → numeric output).
        # Only drops examples where BOTH sides have almost no Urdu.
        if source in ("urdu-instruct", "aya-urdu"):
            instr_ratio = urdu_ratio(ex["instruction"])
            out_ratio = urdu_ratio(ex["output"])
            best_ratio = max(instr_ratio, out_ratio)
            if best_ratio < 0.3:
                reasons["low_urdu_ratio"] += 1
                removed.append({**ex, "removal_reason": f"low_urdu_ratio (instr:{instr_ratio:.0%}, out:{out_ratio:.0%})"})
                continue

        # Passed all checks
        kept.append(ex)

    # Write cleaned data
    log.info(f"Writing {len(kept):,} cleaned examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write removed examples for inspection
    log.info(f"Writing {len(removed):,} removed examples to {REMOVED_FILE}...")
    with open(REMOVED_FILE, "w", encoding="utf-8") as f:
        for ex in removed:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    source_counts = Counter(ex["source"] for ex in kept)
    stats = {
        "input_total": len(raw),
        "output_total": len(kept),
        "removed_total": len(removed),
        "removal_reasons": dict(reasons),
        "kept_by_source": dict(source_counts),
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    log.info(f"\n{'='*50}")
    log.info("CLEANING SUMMARY")
    log.info(f"{'='*50}")
    log.info(f"  Input:   {len(raw):>7,}")
    log.info(f"  Kept:    {len(kept):>7,} ({100*len(kept)/len(raw):.1f}%)")
    log.info(f"  Removed: {len(removed):>7,} ({100*len(removed)/len(raw):.1f}%)")

    log.info(f"\nRemoval reasons:")
    for reason, count in reasons.most_common():
        log.info(f"  {reason:30s} {count:>6,}")

    log.info(f"\nKept by source:")
    for source, count in source_counts.most_common():
        pct = 100 * count / len(kept)
        log.info(f"  {source:25s} {count:>7,} ({pct:5.1f}%)")

    log.info(f"\nOutput: {OUTPUT_FILE}")
    log.info(f"Removed: {REMOVED_FILE}")
    log.info(f"Stats:  {STATS_FILE}")


if __name__ == "__main__":
    main()
