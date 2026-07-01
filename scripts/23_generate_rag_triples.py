"""
Generate RAG training triples so v3 learns to use retrieved context.

The problem (from Tier 1.2): v2 was trained only on plain Q->A pairs, so at
inference the /rag endpoint's `{chunks}\n\nسوال: {query}` format is
out-of-distribution and the model ignores or is confused by the context
(RAG-FT lost 17/100 vs plain FT). This script produces examples in EXACTLY that
surface form (via src.rag.rag_format) so the model sees it during training.

Three kinds of example, chosen per gold chunk by a weighted draw:
    single (~40%)  — context = [gold chunk]; answer grounded in it.
    multi  (~45%)  — context = [gold + 2 distractor chunks from OTHER articles],
                     gold position shuffled; answer grounded in the gold chunk.
                     Teaches "find the relevant chunk among noise."
    noise  (~15%)  — context = [3 unrelated chunks, gold absent]; the answer
                     says the context is insufficient and DECLINES — it does
                     NOT assert any LLM-generated "general-knowledge" fact (an
                     audit found ~18% of those were wrong/unverifiable, often in
                     sensitive domains). Teaches "ignore irrelevant context
                     instead of forcing it into the answer" — the key behaviour
                     that made RAG hurt non-factual tasks in v2.

Chunks come from re-chunking wikimedia/wikipedia 20231101.ur with the SAME
chunker (src.rag.chunking) used to build the Qdrant index, so a training
context is byte-identical to what retrieval actually serves.

Every generated question is screened against the 100 eval prompts
(src.data.eval_guard) so the test set can't leak into training.

Run with:
    python scripts/23_generate_rag_triples.py --n 1800 --budget 2.50
Resume:
    python scripts/23_generate_rag_triples.py --n 1800 --budget 2.50 --resume
Smoke:
    python scripts/23_generate_rag_triples.py --n 5 --budget 0.20 --max-articles 300
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI

# Local modules — canonical surface form + eval guard.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.chunking import chunk_text
from src.rag.rag_format import RAG_SYSTEM, build_rag_user
from src.data.eval_guard import load_eval_guard, _normalize

load_dotenv()

# ─── Config ───
MODEL = "gpt-4o-mini"
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60
BATCH_SIZE = 3            # gold chunks per API call
SLEEP_BETWEEN = 0.5
TEMPERATURE = 0.4         # low — grounded, factual generation

DATASET = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.ur"

# Keep chunks that comfortably fit 3-up under the 4096 train seq-len.
MIN_CHUNK_TOKENS = 120
MAX_CHUNK_TOKENS = 460
DISTRACTOR_POOL_CAP = 800

# kind -> weight (single / multi / noise)
KIND_WEIGHTS = {"single": 0.40, "multi": 0.45, "noise": 0.15}

OUTPUT_DIR = Path("data/rag_triples")
OUTPUT_FILE = OUTPUT_DIR / "rag_triples.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

SYSTEM_PROMPT = (
    "You build Urdu reading-comprehension data from Wikipedia text chunks. "
    "For each chunk, produce ONE natural Urdu question that can be answered "
    "using ONLY that chunk, and a concise Urdu answer that uses ONLY facts "
    "stated in the chunk (no outside facts, 1-3 sentences, no repetition). If "
    "the chunk is too short, listy, or unsuitable for a factual question, set "
    "answerable=false for that item. "
    'Return ONLY JSON: {"examples":[{"id":<int>,"answerable":<bool>,'
    '"question":"<urdu>","grounded_answer":"<urdu>"}]}'
)

# Noise-kind gold answers: DECLINE only — never assert an LLM-generated fact.
# Varied phrasings so the model learns the behaviour, not one canned string.
NOISE_DECLINES = [
    "دیے گئے سیاق میں اس سوال کا جواب موجود نہیں۔",
    "فراہم کردہ متن میں اس سوال کا جواب موجود نہیں۔ درست معلومات کے لیے کسی مستند ماخذ سے رجوع کریں۔",
    "اس سوال کا جواب دیے گئے سیاق میں نہیں ملتا۔",
    "دیے گئے سیاق میں اس کا واضح جواب موجود نہیں؛ بہتر ہے کسی مستند ذریعے سے تصدیق کی جائے۔",
    "فراہم کردہ سیاق اس سوال کا جواب دینے کے لیے کافی نہیں۔",
]


def build_user_message(batch):
    """batch: list of {id, title, chunk}. Compact JSON in."""
    items = [{"id": i, "title": b["title"], "chunk": b["chunk"]} for i, b in enumerate(batch)]
    return json.dumps(items, ensure_ascii=False)


def parse_response(text):
    """Return list of example dicts, or None on failure."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(l for l in cleaned.split("\n") if not l.strip().startswith("```"))
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        obj = obj.get("examples")
    return obj if isinstance(obj, list) else None


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"done": 0, "articles_consumed": 0,
            "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}


def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)


def stream_gold_chunks(max_articles, articles_to_skip):
    """Yield (title, chunk_text, all_chunks_of_article) for usable chunks.

    Streams the Urdu Wikipedia, re-chunks each article with the production
    chunker, and yields chunks in the [MIN,MAX] token band. `articles_to_skip`
    lets --resume continue past already-consumed articles.
    """
    ds = load_dataset(DATASET, DATASET_CONFIG, split="train", streaming=True)
    ds = itertools.islice(ds, articles_to_skip, max_articles)
    for art_idx, row in enumerate(ds, start=articles_to_skip):
        title = (row.get("title") or "").strip()
        chunks = chunk_text(row.get("text") or "")
        usable = [c.text for c in chunks if MIN_CHUNK_TOKENS <= c.approx_tokens <= MAX_CHUNK_TOKENS]
        for ch in usable:
            yield art_idx + 1, title, ch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1800, help="Target usable triples")
    parser.add_argument("--budget", type=float, default=2.50, help="Max spend ($)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-articles", type=int, default=40000,
                        help="Cap on articles streamed (bounds the run)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    guard = load_eval_guard()
    rng = random.Random(args.seed)

    # Resume: rebuild seen-questions + counts from existing output.
    seen_questions: set[str] = set()
    kind_counts = {"single": 0, "multi": 0, "noise": 0}
    if args.resume and OUTPUT_FILE.exists():
        progress = load_progress()
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen_questions.add(rec.get("_qnorm", ""))
                kind_counts[rec.get("rag_kind", "single")] = kind_counts.get(rec.get("rag_kind", "single"), 0) + 1
        print(f"Resuming: {progress['done']} triples, "
              f"{progress['articles_consumed']} articles consumed, "
              f"${progress['total_cost']:.4f} spent")
    else:
        progress = {"done": 0, "articles_consumed": 0,
                    "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    distractor_pool: list[tuple[str, str]] = []  # (title, chunk)
    batch: list[dict] = []
    skipped_contam = skipped_dup = skipped_unanswerable = failed = 0

    print(f"\nPlan: target {args.n} triples | budget ${args.budget:.2f} | model {MODEL}")
    print(f"Kinds: single {KIND_WEIGHTS['single']:.0%} / multi {KIND_WEIGHTS['multi']:.0%} / noise {KIND_WEIGHTS['noise']:.0%}\n")

    gold_stream = stream_gold_chunks(args.max_articles, progress["articles_consumed"])

    def flush_batch():
        """Send one batch of gold chunks, write resulting triples."""
        nonlocal skipped_contam, skipped_dup, skipped_unanswerable, failed
        if not batch:
            return
        user_msg = build_user_message(batch)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                temperature=TEMPERATURE,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print(f"  API error: {e}")
            failed += 1
            time.sleep(2)
            batch.clear()
            return

        u = resp.usage
        progress["total_input_tokens"] += u.prompt_tokens
        progress["total_output_tokens"] += u.completion_tokens
        progress["total_cost"] += (u.prompt_tokens * INPUT_COST_PER_M
                                   + u.completion_tokens * OUTPUT_COST_PER_M) / 1_000_000

        parsed = parse_response(resp.choices[0].message.content)
        if parsed is None:
            failed += 1
            batch.clear()
            return

        for item in parsed:
            idx = item.get("id")
            if not isinstance(idx, int) or idx >= len(batch):
                continue
            gold = batch[idx]
            if not item.get("answerable", False):
                skipped_unanswerable += 1
                continue
            question = (item.get("question") or "").strip()
            grounded = (item.get("grounded_answer") or "").strip()
            if not question or not grounded:
                skipped_unanswerable += 1
                continue

            # Eval-contamination + dedup screens.
            if guard.is_contaminated(question, title=gold["title"]):
                skipped_contam += 1
                continue
            qnorm = " ".join(_normalize(question).split())
            if qnorm in seen_questions:
                skipped_dup += 1
                continue
            seen_questions.add(qnorm)

            # Pick a kind, keeping the running mix near KIND_WEIGHTS.
            kind = rng.choices(list(KIND_WEIGHTS), weights=list(KIND_WEIGHTS.values()))[0]

            distractors = [c for (t, c) in distractor_pool if t != gold["title"]]
            if kind in ("multi", "noise") and len(distractors) < (2 if kind == "multi" else 3):
                kind = "single"  # not enough distractors yet (early in the run)

            if kind == "single":
                ctx = [gold["chunk"]]
                answer = grounded
            elif kind == "multi":
                picks = rng.sample(distractors, 2)
                ctx = [gold["chunk"], *picks]
                rng.shuffle(ctx)  # gold not always first
                answer = grounded
            else:  # noise
                ctx = rng.sample(distractors, 3)
                # Decline only — the gold NEVER asserts an LLM "general
                # knowledge" fact (unverified confabulation vector; an audit
                # found ~18% wrong/unverifiable, many in sensitive domains).
                answer = rng.choice(NOISE_DECLINES)

            record = {
                "system": RAG_SYSTEM,
                "instruction": build_rag_user(ctx, question),
                "input": "",
                "output": answer,
                "source": "rag-grounded-synthetic",
                "rag_kind": kind,
                "gold_title": None if kind == "noise" else gold["title"],
                "n_context_chunks": len(ctx),
                "_qnorm": qnorm,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress["done"] += 1
            kind_counts[kind] += 1

            # Add this gold chunk to the distractor pool for later examples.
            distractor_pool.append((gold["title"], gold["chunk"]))
            if len(distractor_pool) > DISTRACTOR_POOL_CAP:
                distractor_pool.pop(0)

        batch.clear()
        out_f.flush()
        save_progress(progress)

    # ─── Main loop ───
    for art_no, title, chunk in gold_stream:
        progress["articles_consumed"] = art_no
        batch.append({"title": title, "chunk": chunk})

        if len(batch) >= BATCH_SIZE:
            flush_batch()
            time.sleep(SLEEP_BETWEEN)

            if progress["done"] >= args.n:
                break
            if progress["total_cost"] >= args.budget:
                print(f"\n  BUDGET REACHED: ${progress['total_cost']:.4f} >= ${args.budget:.2f}")
                break
            if (progress["done"] and progress["done"] % 30 == 0):
                print(f"  {progress['done']}/{args.n} | "
                      f"single {kind_counts['single']} multi {kind_counts['multi']} noise {kind_counts['noise']} | "
                      f"${progress['total_cost']:.4f} | art {progress['articles_consumed']:,}")

    flush_batch()  # trailing partial batch
    out_f.close()

    print(f"\n{'='*50}\nDONE\n{'='*50}")
    print(f"Triples written:   {progress['done']:,}")
    print(f"  single/multi/noise: {kind_counts['single']}/{kind_counts['multi']}/{kind_counts['noise']}")
    print(f"Skipped — contam {skipped_contam}, dup {skipped_dup}, unanswerable {skipped_unanswerable}, failed batches {failed}")
    print(f"Articles consumed: {progress['articles_consumed']:,}")
    print(f"Total cost:        ${progress['total_cost']:.4f}")
    print(f"Output:            {OUTPUT_FILE}")
    print(f"\nResume: python scripts/23_generate_rag_triples.py --n {args.n} --budget {args.budget} --resume")


if __name__ == "__main__":
    main()
