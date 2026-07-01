"""
Generate Urdu summarization + reasoning data so v3 recovers two v2 regressions.

v2 eval regressions this targets:
    summarization 73% -> 46%   (model stopped honoring explicit length/format)
    reasoning     44% -> 31%   (model hallucinated arithmetic)

Both categories regressed for the same underlying reason: v2's data had little
supervision for "follow this exact constraint" and "the numbers must be right."
This script builds that supervision, with a self-checking trick for each kind so
gpt-4o-mini can't introduce the very failure we're trying to fix.

  SUMMARIZATION (kind="summarization")
    Each example asks for a summary in an EXPLICIT number of sentences
    ("اس متن کا تین جملوں میں خلاصہ کریں"). After generation we COUNT the Urdu
    sentences in the summary and keep the example only if the count matches
    exactly. The passage is real Urdu Wikipedia (streamed + re-chunked with the
    production chunker), so the source text is always natural. One corrective
    retry is allowed before a mismatch is discarded.

  REASONING (kind="reasoning")
    Python owns ALL the arithmetic. We generate a templated word problem with
    random numbers and compute the exact answer + every intermediate value in
    Python. gpt-4o-mini only PHRASES the problem and NARRATES the pre-computed
    steps in natural Urdu/Roman — it never does math. We then verify that the
    Python answer and intermediates actually appear (as ASCII digits) in the
    model's output, and discard otherwise. The LLM can't hallucinate a number
    into the gold, because the gold numbers are fixed before it ever runs.
    ~25% of reasoning examples are Roman-Urdu (numbers stay ASCII either way).

Every passage / problem is screened against the 100 eval prompts
(src.data.eval_guard) so the summarization + reasoning test sets can't leak in.

Run with:
    python scripts/25_generate_summ_reason.py --n 2000 --budget 1.50
Resume:
    python scripts/25_generate_summ_reason.py --n 2000 --budget 1.50 --resume
Smoke (tiny spend):
    python scripts/25_generate_summ_reason.py --n-summ 3 --n-reason 3 --budget 0.10 --max-articles 400
Self-test (free, no API):
    python scripts/25_generate_summ_reason.py --self-test
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.chunking import chunk_text, split_sentences
from src.data.eval_guard import load_eval_guard, _normalize

load_dotenv()

# ─── Shared config ───
MODEL = "gpt-4o-mini"
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60
SLEEP_BETWEEN = 0.5

DATASET = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.ur"

OUTPUT_DIR = Path("data/summ_reason")
OUTPUT_FILE = OUTPUT_DIR / "summ_reason.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Urdu words for small sentence counts (matches eval prompts: "تین جملوں میں").
NUM_WORDS_URDU = {1: "ایک", 2: "دو", 3: "تین", 4: "چار", 5: "پانچ"}

SENT_END = re.compile(r"[۔؟!]+")
EASTERN_DIGITS = re.compile(r"[۰-۹٠-٩]")
URDU_LETTERS = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LATIN_LETTERS = re.compile(r"[A-Za-z]")


def script_ok(text: str, lang: str) -> bool:
    """Output script must match the requested lang (digits/symbols ignored).
    Catches the model transliterating an Urdu item into Roman, or vice versa."""
    u = len(URDU_LETTERS.findall(text))
    a = len(LATIN_LETTERS.findall(text))
    tot = u + a
    if tot == 0:
        return False
    if lang == "urdu":
        return u / tot >= 0.5
    return a / tot >= 0.6 and u / tot <= 0.1


def count_sentences(text: str) -> int:
    """Count Urdu sentences by terminator (۔ ؟ !). ASCII '.' is NOT a terminator
    here, so decimals like '3.5' don't inflate the count."""
    return len([p for p in SENT_END.split(text) if p.strip()])


def numbers_in(text: str) -> set[int]:
    """Set of integers appearing in text, with digit-grouping commas removed so
    '2,040' reads as 2040."""
    t = text.replace(",", "").replace("٬", "").replace("،", "")
    return {int(x) for x in re.findall(r"\d+", t)}


def parse_items(text: str, key: str) -> list | None:
    """Return the list under `key`, tolerating ``` fences. None on failure."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(l for l in cleaned.split("\n") if not l.strip().startswith("```"))
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        obj = obj.get(key)
    return obj if isinstance(obj, list) else None


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return _fresh_progress()


def _fresh_progress() -> dict:
    return {"summ_done": 0, "reason_done": 0, "reason_roman": 0,
            "articles_consumed": 0,
            "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}


def save_progress(p: dict) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)


def _account(progress: dict, usage) -> None:
    progress["total_input_tokens"] += usage.prompt_tokens
    progress["total_output_tokens"] += usage.completion_tokens
    progress["total_cost"] += (usage.prompt_tokens * INPUT_COST_PER_M
                               + usage.completion_tokens * OUTPUT_COST_PER_M) / 1_000_000


# ════════════════════════════ SUMMARIZATION ════════════════════════════

SUMM_MIN_TOK = 120
SUMM_MAX_TOK = 400
SUMM_MIN_SENTS = 4          # need enough source sentences to compress

SUMM_SYSTEM_PROMPT = (
    "You write faithful Urdu summaries with an EXACT sentence count. For each "
    "item you get an Urdu passage and target_sentences (an integer). Write a "
    "summary in Urdu that:\n"
    "  - contains EXACTLY target_sentences sentences, each ending with the Urdu "
    "full stop \"۔\",\n"
    "  - covers only the main points actually stated in the passage (no outside "
    "facts, no opinions, no repetition),\n"
    "  - is fluent natural Urdu, NOT a word-for-word copy of the passage.\n"
    "Produce neither more nor fewer than target_sentences sentences.\n"
    'Return ONLY JSON: {"summaries":[{"id":<int>,"summary":"<urdu>"}]}'
)

SUMM_INSTRUCTIONS = [
    "اس متن کا {n} جملوں میں خلاصہ کریں:\n\n{passage}",
    "درج ذیل عبارت کا خلاصہ {n} جملوں میں لکھیں:\n\n{passage}",
    "مندرجہ ذیل متن کا {n} جملوں میں مختصر خلاصہ دیں:\n\n{passage}",
]


def choose_target_sentences(n_src: int, rng: random.Random) -> int:
    """Pick a target sentence count that is a genuine compression of the source
    and within easy reach (matches eval prompts asking for 2-3 sentences)."""
    if n_src <= 3:
        lo, hi = 1, 2
    elif n_src <= 5:
        lo, hi = 2, 3
    elif n_src <= 8:
        lo, hi = 2, 4
    else:
        lo, hi = 3, 5
    hi = min(hi, max(1, n_src - 1))
    lo = min(lo, hi)
    return rng.randint(lo, hi)


def stream_summ_passages(max_articles: int, skip: int):
    """Yield (art_no, title, passage) for Wikipedia chunks usable as summary
    source text. `datasets` is imported lazily so reasoning-only/self-test runs
    don't pay for it."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, DATASET_CONFIG, split="train", streaming=True)
    ds = itertools.islice(ds, skip, max_articles)
    for art_idx, row in enumerate(ds, start=skip):
        title = (row.get("title") or "").strip()
        for c in chunk_text(row.get("text") or ""):
            if SUMM_MIN_TOK <= c.approx_tokens <= SUMM_MAX_TOK and count_sentences(c.text) >= SUMM_MIN_SENTS:
                yield art_idx + 1, title, c.text


def _summ_user(items: list[dict], corrective: bool) -> str:
    """items: list of {passage, target, [got]}. Compact JSON; id = position."""
    if corrective:
        payload = [{"id": i, "passage": it["passage"], "target_sentences": it["target"],
                    "note": f"your previous summary had {it['got']} sentences; "
                            f"produce EXACTLY {it['target']}"} for i, it in enumerate(items)]
    else:
        payload = [{"id": i, "passage": it["passage"], "target_sentences": it["target"]}
                   for i, it in enumerate(items)]
    return json.dumps(payload, ensure_ascii=False)


def generate_summarization(client, guard, rng, out_f, seen, progress, args, counters) -> None:
    SUMM_BATCH = 5
    batch: list[dict] = []

    def summ_call(items, corrective):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SUMM_SYSTEM_PROMPT},
                          {"role": "user", "content": _summ_user(items, corrective)}],
                temperature=0.4,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print(f"  API error (summ): {e}")
            counters["failed"] += 1
            time.sleep(2)
            return None
        _account(progress, resp.usage)
        parsed = parse_items(resp.choices[0].message.content, "summaries")
        if parsed is None:
            counters["failed"] += 1
            return None
        out = {}
        for it in parsed:
            i = it.get("id")
            if isinstance(i, int) and 0 <= i < len(items):
                out[i] = (it.get("summary") or "").strip()
        return out

    def accept(passage, summary, target, title):
        # Not a verbatim copy of the passage.
        if " ".join(_normalize(summary).split()) == " ".join(_normalize(passage).split()):
            return False
        n_word = NUM_WORDS_URDU.get(target, str(target))
        record = {
            "instruction": rng.choice(SUMM_INSTRUCTIONS).format(n=n_word, passage=passage),
            "input": "",
            "output": summary,
            "source": "summ-synthetic",
            "kind": "summarization",
            "n_sentences": target,
            "passage_title": title,
            "_qnorm": " ".join(_normalize(passage).split()),
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        progress["summ_done"] += 1
        return True

    def flush():
        if not batch:
            return
        first = summ_call(batch, corrective=False)
        if first is None:
            batch.clear()
            return
        retry: list[dict] = []
        accepted_idx = set()
        for i, it in enumerate(batch):
            s = first.get(i, "")
            if s and count_sentences(s) == it["target"]:
                if accept(it["passage"], s, it["target"], it["title"]):
                    accepted_idx.add(i)
                else:
                    counters["copy"] += 1
            else:
                it["got"] = count_sentences(s) if s else 0
                retry.append(it)
        # One corrective retry for count mismatches.
        if retry:
            second = summ_call(retry, corrective=True)
            if second is not None:
                for j, it in enumerate(retry):
                    s = second.get(j, "")
                    if s and count_sentences(s) == it["target"] and accept(
                            it["passage"], s, it["target"], it["title"]):
                        pass
                    else:
                        counters["count_mismatch"] += 1
            else:
                counters["count_mismatch"] += len(retry)
        batch.clear()
        out_f.flush()
        save_progress(progress)

    print(f"\n[summarization] target {args.n_summ} | from Wikipedia stream")
    stream = stream_summ_passages(args.max_articles, progress["articles_consumed"])
    for art_no, title, passage in stream:
        progress["articles_consumed"] = art_no
        n_src = count_sentences(passage)
        if n_src < SUMM_MIN_SENTS:
            continue
        qnorm = " ".join(_normalize(passage).split())
        if qnorm in seen:
            counters["dup"] += 1
            continue
        if guard.is_contaminated(passage, title=title):
            counters["contam"] += 1
            continue
        seen.add(qnorm)
        batch.append({"passage": passage, "target": choose_target_sentences(n_src, rng), "title": title})
        if len(batch) >= SUMM_BATCH:
            flush()
            time.sleep(SLEEP_BETWEEN)
            if progress["summ_done"] >= args.n_summ:
                break
            if progress["total_cost"] >= args.budget:
                print(f"  BUDGET REACHED in summarization: ${progress['total_cost']:.4f}")
                break
            if progress["summ_done"] and progress["summ_done"] % 40 == 0:
                print(f"  summ {progress['summ_done']}/{args.n_summ} | "
                      f"${progress['total_cost']:.4f} | art {progress['articles_consumed']:,}")
    flush()


# ════════════════════════════ REASONING ════════════════════════════
# Varied surface details (names, items) keep generated problems from collapsing
# onto the eval prompts' exact wording — better generalization, fewer eval-guard
# collisions.
NAMES = ["Ahmad", "Bilal", "Sara", "Ayesha", "Hamza", "Zara", "Usman",
         "Fatima", "Omar", "Hina", "Imran", "Nida", "Tariq", "Saad"]
ITEMS = ["books", "notebooks", "pens", "copies", "files", "markers", "folders"]
EXTRAS = ["pen", "eraser", "ruler", "sharpener", "bag"]

# Python computes everything; the LLM only narrates. Each template returns:
#   name, params, input_numbers (must appear in problem), intermediates +
#   final_answer (must appear in solution), scenario_en/question_en (for the LLM
#   to render), steps (ascii-math narration the LLM rephrases).

def _t_profit_resale(rng):
    buy1 = rng.choice(range(200, 2001, 50)); m1 = rng.choice([100, 150, 200, 250, 300]); sell1 = buy1 + m1
    buy2 = sell1 + rng.choice([50, 100, 150]); m2 = rng.choice([100, 150, 200, 250, 300]); sell2 = buy2 + m2
    p1, p2 = sell1 - buy1, sell2 - buy2
    total = p1 + p2
    return {
        "name": "profit_resale",
        "params": {"buy1": buy1, "sell1": sell1, "buy2": buy2, "sell2": sell2},
        "input_numbers": [buy1, sell1, buy2, sell2], "intermediates": [p1, p2], "final_answer": total,
        "scenario_en": f"A shopkeeper bought a shirt for {buy1} rupees and sold it for {sell1} rupees. "
                       f"Later he bought the same shirt again for {buy2} rupees and sold it for {sell2} rupees.",
        "question_en": "What is his total profit in rupees?",
        "steps": [f"first profit = {sell1} - {buy1} = {p1}",
                  f"second profit = {sell2} - {buy2} = {p2}",
                  f"total profit = {p1} + {p2} = {total}"],
    }


def _t_speed_time(rng):
    speed = rng.choice([40, 50, 60, 80, 100, 120]); hours = rng.choice([2, 3, 4, 5, 6, 7]); dist = speed * hours
    return {
        "name": "speed_time",
        "params": {"speed": speed, "hours": hours, "dist": dist},
        "input_numbers": [dist, speed], "intermediates": [], "final_answer": hours,
        "scenario_en": f"A train travels at {speed} km per hour and must cover a distance of {dist} km.",
        "question_en": "How many hours will the journey take?",
        "steps": [f"time = distance / speed", f"time = {dist} / {speed} = {hours}", f"answer = {hours} hours"],
    }


def _t_age_sum(rng):
    a_name, b_name, c_name = rng.sample(NAMES, 3)
    bilal = rng.randint(10, 40); da = rng.randint(2, 6); dk = rng.randint(1, 5)
    ahmad = bilal + da; kamran = bilal - dk
    total = ahmad + bilal + kamran
    return {
        "name": "age_sum",
        "params": {"bilal": bilal, "da": da, "dk": dk, "ahmad": ahmad, "kamran": kamran},
        "input_numbers": [bilal, da, dk], "intermediates": [ahmad, kamran], "final_answer": total,
        "scenario_en": f"Among three friends, {b_name} is {bilal} years old. {a_name} is {da} years older "
                       f"than {b_name}, and {c_name} is {dk} years younger than {b_name}.",
        "question_en": "What is the sum of their three ages?",
        "steps": [f"{a_name} = {bilal} + {da} = {ahmad}", f"{c_name} = {bilal} - {dk} = {kamran}",
                  f"total = {ahmad} + {bilal} + {kamran} = {total}"],
    }


def _t_total_cost(rng):
    name = rng.choice(NAMES); item = rng.choice(ITEMS); extra = rng.choice(EXTRAS)
    qty = rng.randint(2, 6); unit = rng.choice(range(20, 201, 10)); pen = rng.choice(range(20, 101, 10))
    books = qty * unit; spent = books + pen; start = spent + rng.choice(range(50, 501, 50))
    left = start - spent
    return {
        "name": "total_cost",
        "params": {"qty": qty, "unit": unit, "pen": pen, "start": start},
        "input_numbers": [start, qty, unit, pen], "intermediates": [books, spent], "final_answer": left,
        "scenario_en": f"{name} has {start} rupees and buys {qty} {item} at {unit} rupees each, then buys "
                       f"a {extra} for {pen} rupees.",
        "question_en": f"How many rupees does {name} have left?",
        "steps": [f"cost of {item} = {qty} * {unit} = {books}", f"total spent = {books} + {pen} = {spent}",
                  f"remaining = {start} - {spent} = {left}"],
    }


def _t_savings_year(rng):
    daily = rng.choice([50, 100, 150, 200, 250]); total = daily * 365
    return {
        "name": "savings_year",
        "params": {"daily": daily}, "input_numbers": [daily], "intermediates": [365], "final_answer": total,
        "scenario_en": f"A person saves {daily} rupees every day for one full year.",
        "question_en": "How many rupees will be saved in a year?",
        "steps": [f"one year = 365 days", f"total = {daily} * 365 = {total}", f"answer = {total} rupees"],
    }


def _t_capacity_rooms(rng):
    rooms = rng.randint(3, 8); per = rng.choice([25, 30, 35, 40]); extra = rng.choice([1, 2, 3])
    cap = rooms * per; students = cap + extra * per; short = students - cap
    return {
        "name": "capacity_rooms",
        "params": {"rooms": rooms, "per": per, "students": students},
        "input_numbers": [rooms, per, students], "intermediates": [cap, short], "final_answer": extra,
        "scenario_en": f"A school has {rooms} rooms and each room seats {per} students. {students} "
                       f"students want admission.",
        "question_en": "How many additional rooms are needed to seat everyone?",
        "steps": [f"capacity = {rooms} * {per} = {cap}", f"students without seats = {students} - {cap} = {short}",
                  f"extra rooms = {short} / {per} = {extra}"],
    }


def _t_trees_fruit(rng):
    houses = rng.randint(8, 25); tph = rng.choice([2, 3]); fpt = rng.choice([10, 15, 20, 25])
    trees = houses * tph; fruit = trees * fpt
    return {
        "name": "trees_fruit",
        "params": {"houses": houses, "tph": tph, "fpt": fpt},
        "input_numbers": [houses, tph, fpt], "intermediates": [trees], "final_answer": fruit,
        "scenario_en": f"In a neighborhood there are {houses} houses, and {tph} trees are planted in front "
                       f"of each house. Each tree gives {fpt} kg of fruit per year.",
        "question_en": "How many kg of fruit do all the trees give in one year?",
        "steps": [f"total trees = {houses} * {tph} = {trees}", f"total fruit = {trees} * {fpt} = {fruit}",
                  f"answer = {fruit} kg"],
    }


def _t_discount(rng):
    price = rng.choice(range(500, 5001, 100)); pct = rng.choice([5, 10, 15, 20, 25])
    disc = price * pct // 100; final = price - disc
    return {
        "name": "discount",
        "params": {"price": price, "pct": pct},
        "input_numbers": [price, pct], "intermediates": [disc], "final_answer": final,
        "scenario_en": f"A jacket costs {price} rupees and the shop offers a {pct} percent discount.",
        "question_en": "What is the final price in rupees?",
        "steps": [f"discount = {price} * {pct} / 100 = {disc}", f"final price = {price} - {disc} = {final}"],
    }


TEMPLATES = [_t_profit_resale, _t_speed_time, _t_age_sum, _t_total_cost,
             _t_savings_year, _t_capacity_rooms, _t_trees_fruit, _t_discount]


def make_problem(rng: random.Random) -> dict:
    return rng.choice(TEMPLATES)(rng)


# Python owns the QUESTION (as it owns the arithmetic). The model renders only
# the scenario; we append the canonical question per template so it can NEVER be
# dropped — the age_sum failure mode that was 4/4 fails in the v3 spot-check.
# Values are (urdu, roman) per template name.
REASON_QUESTIONS = {
    "profit_resale": ("اس کا کل منافع کتنے روپے ہے؟", "Iska kul munafa kitne rupay hai?"),
    "speed_time": ("سفر میں کتنے گھنٹے لگیں گے؟", "Safar mein kitne ghante lagenge?"),
    "age_sum": ("ان تینوں کی عمروں کا مجموعہ کتنا ہے؟", "In teenon ki umron ka majmua kitna hai?"),
    "total_cost": ("اس کے پاس کتنے روپے بچے؟", "Uske paas kitne rupay bache?"),
    "savings_year": ("ایک سال میں کتنے روپے جمع ہوں گے؟", "Ek saal mein kitne rupay jama honge?"),
    "capacity_rooms": ("سب کو بٹھانے کے لیے کتنے اضافی کمرے درکار ہیں؟",
                       "Sab ko bithane ke liye kitne izafi kamre darkar hain?"),
    "trees_fruit": ("تمام درختوں سے ایک سال میں کتنے کلو پھل ملتے ہیں؟",
                    "Tamam darakhton se ek saal mein kitne kilo phal milte hain?"),
    "discount": ("آخری قیمت کتنے روپے ہے؟", "Aakhri qeemat kitne rupay hai?"),
}


REASON_SYSTEM_PROMPT = (
    "You phrase pre-solved math word problems in fluent natural language. Each "
    "item has: lang, a scenario (in English), and solved steps.\n"
    "SCRIPT — obey the item's lang exactly:\n"
    "  lang=\"urdu\": write BOTH problem and solution in URDU (Nastaliq) script — "
    "translate the English scenario into natural Urdu. Use NO Latin/Roman letters "
    "at all, except the ASCII digits 0-9.\n"
    "  lang=\"roman\": write in Roman-Urdu — Latin letters, the way Pakistanis "
    "type on WhatsApp.\n"
    "Produce, in that script:\n"
    "  - problem: state the SCENARIO naturally, using EXACTLY the given numbers. "
    "Do NOT add a question and do NOT solve it — a question is appended later.\n"
    "  - solution: narrate the given steps in order and end with a clear sentence "
    "stating the final answer. Use EXACTLY the given numbers and results.\n"
    "RULES: never invent, change, drop, or recompute a number — only restate the "
    "values you are given. Write ALL digits as ASCII numerals (e.g. 2040, never "
    "۲۰۴۰). Keep it concise and correct.\n"
    'Return ONLY JSON: {"items":[{"id":<int>,"problem":"...","solution":"..."}]}'
)


def _reason_user(batch: list[dict]) -> str:
    payload = [{"id": i, "lang": f["lang"], "scenario": f["scenario_en"],
                "steps": f["steps"], "final_answer": f["final_answer"]}
               for i, f in enumerate(batch)]
    return json.dumps(payload, ensure_ascii=False)


def validate_reason(problem: str, solution: str, facts: dict) -> bool:
    problem = (problem or "").strip()
    solution = (solution or "").strip()
    if not problem or not solution:
        return False
    if EASTERN_DIGITS.search(problem) or EASTERN_DIGITS.search(solution):
        return False  # must stay ASCII to match the eval surface form
    if not script_ok(problem, facts["lang"]) or not script_ok(solution, facts["lang"]):
        return False  # model transliterated instead of honoring lang
    pnums = numbers_in(problem)
    if any(n not in pnums for n in facts["input_numbers"]):
        return False  # a problem number was dropped or altered
    snums = numbers_in(solution)
    if facts["final_answer"] not in snums:
        return False  # the Python answer must be stated
    if any(n not in snums for n in facts["intermediates"]):
        return False  # an intermediate step value is missing
    return True


def generate_reasoning(client, guard, rng, out_f, seen, progress, args, counters) -> None:
    REASON_BATCH = 5
    batch: list[dict] = []

    def flush():
        if not batch:
            return
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": REASON_SYSTEM_PROMPT},
                          {"role": "user", "content": _reason_user(batch)}],
                temperature=0.4,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print(f"  API error (reason): {e}")
            counters["failed"] += 1
            time.sleep(2)
            batch.clear()
            return
        _account(progress, resp.usage)
        parsed = parse_items(resp.choices[0].message.content, "items")
        if parsed is None:
            counters["failed"] += 1
            batch.clear()
            return
        for it in parsed:
            i = it.get("id")
            if not isinstance(i, int) or i >= len(batch):
                continue
            facts = batch[i]
            problem = (it.get("problem") or "").strip()
            solution = (it.get("solution") or "").strip()
            # Python owns the question: append the canonical one so the model
            # can't drop it (the age_sum failure mode in the v3 spot-check).
            if problem:
                q_ur, q_roman = REASON_QUESTIONS[facts["name"]]
                problem = problem + " " + (q_roman if facts["lang"] == "roman" else q_ur)
            if not validate_reason(problem, solution, facts):
                counters["bad_numbers"] += 1
                continue
            if guard.is_contaminated(problem, title=None):
                counters["contam"] += 1
                continue
            qnorm = " ".join(_normalize(problem).split())
            if qnorm in seen:
                counters["dup"] += 1
                continue
            seen.add(qnorm)
            record = {
                "instruction": problem,
                "input": "",
                "output": solution,
                "source": "reason-synthetic",
                "kind": "reasoning",
                "template_name": facts["name"],
                "final_answer": str(facts["final_answer"]),
                "is_roman": facts["lang"] == "roman",
                "_qnorm": qnorm,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress["reason_done"] += 1
            if facts["lang"] == "roman":
                progress["reason_roman"] += 1
        batch.clear()
        out_f.flush()
        save_progress(progress)

    print(f"\n[reasoning] target {args.n_reason} | {args.roman_frac:.0%} Roman | Python-grounded")
    attempts = 0
    cap = args.n_reason * 8  # safety bound if validation keeps failing
    while progress["reason_done"] < args.n_reason and attempts < cap:
        f = make_problem(rng)
        # Count in-flight batch too, else a whole batch is decided before any
        # pick updates reason_roman (mix lurches all-Roman then all-Urdu).
        roman_so_far = progress["reason_roman"] + sum(1 for b in batch if b["lang"] == "roman")
        total_so_far = progress["reason_done"] + len(batch)
        want_roman = (roman_so_far / max(1, total_so_far)) < args.roman_frac
        f["lang"] = "roman" if want_roman else "urdu"
        batch.append(f)
        attempts += 1
        if len(batch) >= REASON_BATCH:
            flush()
            time.sleep(SLEEP_BETWEEN)
            if progress["total_cost"] >= args.budget:
                print(f"  BUDGET REACHED in reasoning: ${progress['total_cost']:.4f}")
                break
            if progress["reason_done"] and progress["reason_done"] % 40 == 0:
                print(f"  reason {progress['reason_done']}/{args.n_reason} | "
                      f"roman {progress['reason_roman']} | ${progress['total_cost']:.4f}")
    flush()


# ════════════════════════════ MAIN ════════════════════════════

def run(args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    guard = load_eval_guard()
    rng = random.Random(args.seed)

    seen: set[str] = set()
    if args.resume and OUTPUT_FILE.exists():
        progress = load_progress()
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(rec.get("_qnorm", ""))
        print(f"Resuming: summ {progress['summ_done']}, reason {progress['reason_done']}, "
              f"art {progress['articles_consumed']}, ${progress['total_cost']:.4f} spent")
    else:
        progress = _fresh_progress()
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    counters = {"failed": 0, "contam": 0, "dup": 0, "copy": 0,
                "count_mismatch": 0, "bad_numbers": 0}

    print(f"\nPlan: summ {args.n_summ} + reason {args.n_reason} | budget ${args.budget:.2f} | model {MODEL}")

    # Reasoning first (cheaper, addresses the worse 31% regression) so it's
    # covered even under a tight budget; then summarization.
    if progress["reason_done"] < args.n_reason and progress["total_cost"] < args.budget:
        generate_reasoning(client, guard, rng, out_f, seen, progress, args, counters)
    if progress["summ_done"] < args.n_summ and progress["total_cost"] < args.budget:
        generate_summarization(client, guard, rng, out_f, seen, progress, args, counters)

    out_f.close()
    save_progress(progress)

    print(f"\n{'='*50}\nDONE\n{'='*50}")
    print(f"Summarization written: {progress['summ_done']:,}")
    print(f"Reasoning written:     {progress['reason_done']:,} "
          f"(roman {progress['reason_roman']})")
    print(f"Skipped — contam {counters['contam']}, dup {counters['dup']}, "
          f"copy {counters['copy']}, count-mismatch {counters['count_mismatch']}, "
          f"bad-numbers {counters['bad_numbers']}, failed batches {counters['failed']}")
    print(f"Articles consumed: {progress['articles_consumed']:,}")
    print(f"Total cost: ${progress['total_cost']:.4f}")
    print(f"Output:     {OUTPUT_FILE}")
    print(f"\nResume: python scripts/25_generate_summ_reason.py "
          f"--n-summ {args.n_summ} --n-reason {args.n_reason} --budget {args.budget} --resume")


def _self_test() -> None:
    """Exercise every non-API path on synthetic data. Free, no network."""
    rng = random.Random(0)

    # 1. Sentence counting (Urdu terminators only; ASCII '.' / decimals ignored).
    assert count_sentences("یہ پہلا جملہ ہے۔ یہ دوسرا ہے۔ یہ تیسرا ہے۔") == 3
    assert count_sentences("قیمت 3.5 روپے ہے۔") == 1     # decimal must not split
    assert count_sentences("کیا تم آؤ گے؟ ہاں!") == 2
    print("sentence counting:           OK")

    # 2. numbers_in handles commas; eastern-digit detector works.
    assert numbers_in("قیمت 2,040 اور 360 روپے") == {2040, 360}
    assert EASTERN_DIGITS.search("۸۰۰") and not EASTERN_DIGITS.search("800")
    print("number extraction:           OK")

    # 3. choose_target_sentences always compresses and stays >= 1.
    for n_src in range(2, 20):
        t = choose_target_sentences(n_src, rng)
        assert 1 <= t < n_src, (n_src, t)
    print("target-sentence picker:      OK")

    # 4. EVERY reasoning template: arithmetic is internally correct, the answer
    #    is shown in the last step, and a perfectly-narrated output validates.
    expect = {
        "profit_resale": lambda p: (p["sell1"]-p["buy1"]) + (p["sell2"]-p["buy2"]),
        "speed_time":    lambda p: p["hours"],
        "age_sum":       lambda p: p["ahmad"] + p["bilal"] + p["kamran"],
        "total_cost":    lambda p: p["start"] - (p["qty"]*p["unit"] + p["pen"]),
        "savings_year":  lambda p: p["daily"] * 365,
        "capacity_rooms": None,   # answer is the extra-rooms count, checked structurally
        "trees_fruit":   lambda p: p["houses"] * p["tph"] * p["fpt"],
        "discount":      lambda p: p["price"] - (p["price"]*p["pct"]//100),
    }
    assert set(REASON_QUESTIONS) == {t.__name__.replace("_t_", "") for t in TEMPLATES}, \
        "REASON_QUESTIONS must cover every template"
    seen_names = set()
    for _ in range(2000):
        f = make_problem(rng)
        f["lang"] = "urdu"
        seen_names.add(f["name"])
        fn = expect[f["name"]]
        if fn is not None:
            assert fn(f["params"]) == f["final_answer"], f
        assert str(f["final_answer"]) in f["steps"][-1], f
        assert all(isinstance(n, int) for n in f["input_numbers"] + f["intermediates"])
        # A perfectly-narrated Urdu output (numbers echoed, Urdu filler) plus the
        # Python-appended question validates.
        q_ur = REASON_QUESTIONS[f["name"]][0]
        gp = "سوال " + " ".join(str(n) for n in f["input_numbers"]) + " معلوم کریں۔ " + q_ur
        gs = "حل " + " ".join(str(n) for n in f["intermediates"] + [f["final_answer"]]) + " جواب"
        assert validate_reason(gp, gs, f), f
    assert seen_names == {t.__name__.replace("_t_", "") for t in TEMPLATES}, seen_names
    print(f"reasoning templates:         OK ({len(TEMPLATES)} templates, all hit)")

    # 5. validate_reason rejects: wrong answer, dropped input number, Eastern
    #    digits, and an Urdu item transliterated into Roman (script mismatch).
    f = _t_discount(rng); f["lang"] = "urdu"
    base_p = "قیمت " + " ".join(map(str, f["input_numbers"])) + " روپے"
    assert not validate_reason(base_p, "حل " + str(f["final_answer"] + 1) + " جواب", f)  # wrong answer
    assert not validate_reason("قیمت 100 روپے", "حل " + str(f["final_answer"]) + " جواب", f)  # missing inputs
    assert not validate_reason(base_p, "حل ۸۰۰ جواب", f)                  # eastern digits
    assert not validate_reason("price " + " ".join(map(str, f["input_numbers"])),
                               "ans " + str(f["final_answer"]), f)        # roman for an urdu item
    print("reasoning validation:        OK")

    # 6. Parsers + eval guard load.
    assert parse_items('{"items":[{"id":0,"problem":"p","solution":"s"}]}', "items")[0]["problem"] == "p"
    assert parse_items('{"summaries":[{"id":0,"summary":"x"}]}', "summaries")[0]["summary"] == "x"
    load_eval_guard()
    print("parsers + eval guard:        OK")

    print("\nSELF-TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000,
                        help="Total target; split 50/50 unless --n-summ/--n-reason given")
    parser.add_argument("--n-summ", type=int, default=None, help="Summarization target")
    parser.add_argument("--n-reason", type=int, default=None, help="Reasoning target")
    parser.add_argument("--budget", type=float, default=1.50, help="Max spend ($)")
    parser.add_argument("--roman-frac", type=float, default=0.25,
                        help="Fraction of reasoning examples in Roman-Urdu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-articles", type=int, default=40000,
                        help="Cap on Wikipedia articles streamed for summarization")
    parser.add_argument("--self-test", action="store_true",
                        help="Run free offline checks and exit (no API calls)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.self_test:
        _self_test()
        return

    if args.n_summ is None:
        args.n_summ = args.n // 2
    if args.n_reason is None:
        args.n_reason = args.n - args.n // 2
    run(args)

    # HF `datasets` streaming spawns a background prefetch thread that can crash
    # on interpreter finalization ("PyGILState_Release"). Everything is already
    # flushed + saved by run(), so exit hard to skip the noisy shutdown.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
