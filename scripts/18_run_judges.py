"""Run automated multi-judge evaluation on data/eval/v2/outputs.jsonl.

Three judges via different auth paths:
  - openai  : OpenAI API (gpt-5.3-thinking) — needs OPENAI_API_KEY
  - gemini  : Google AI Studio API (gemini-3.1-pro) — needs GOOGLE_API_KEY
  - claude  : Claude Code CLI subprocess (Opus 4.7 via Max subscription) — no key

Each judge gets the same Jinja-rendered prompt (prompts/judge_v2.j2) with
A/B position randomized per item to control position bias. The judge's
A/B verdict is mapped back to base/finetuned and written in the existing
schema {id, category, winner, reason} consumed by 17_aggregate_judges.py.

Usage:
    python scripts/18_run_judges.py                       # all 3 judges, all items
    python scripts/18_run_judges.py --judges openai gemini --limit 5
    python scripts/18_run_judges.py --resume              # skip already-judged ids

Resume: each judge directory has progress.json with {done_ids: [...]}. Re-running
appends only new items.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from jinja2 import Template

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.eval.judge_pipeline import (  # noqa: E402
    JUDGE_DIR_NAMES,
    build_judge_runner,
    load_judge_template,
)

DEFAULT_OUTPUTS = ROOT / "data" / "eval" / "v2" / "outputs.jsonl"
DEFAULT_JUDGES_DIR = ROOT / "data" / "eval" / "v2" / "judges"
DEFAULT_A_KEY = "base_response"
DEFAULT_B_KEY = "finetuned_response"
REQUIRED_KEYS = {"id", "category", "winner", "reason"}


def winner_label_for(key: str) -> str:
    """Map a response-field name to its winner label.

    e.g. "base_response" -> "base", "finetuned_response" -> "finetuned",
         "rag_response" -> "rag".
    """
    return key.removesuffix("_response")


def render_prompt(template_str: str, *, category: str, prompt: str,
                  response_a: str, response_b: str) -> str:
    return Template(template_str).render(
        category=category,
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )


def parse_judge_reply(text: str) -> dict[str, Any] | None:
    """Extract {winner: A|B|tie, reason: str} from judge reply. Tolerant of markdown."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            l for l in cleaned.split("\n") if not l.strip().startswith("```")
        ).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    winner = obj.get("winner")
    reason = obj.get("reason")
    if winner not in {"A", "B", "tie"} or not isinstance(reason, str):
        return None
    return {"winner": winner, "reason": reason.strip()}


def map_ab_to_labels(verdict_ab: str, a_label: str, b_label: str, a_is_first: bool) -> str:
    """Map A/B verdict back to the original label. `a_is_first` reflects whether
    the response shown in position A came from the FIRST key (a_label)."""
    if verdict_ab == "tie":
        return "tie"
    first_label, second_label = a_label, b_label
    a_side = first_label if a_is_first else second_label
    b_side = second_label if a_is_first else first_label
    return a_side if verdict_ab == "A" else b_side


def load_outputs(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def load_progress(judge_dir: Path) -> set[int]:
    p = judge_dir / "progress.json"
    if not p.exists():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")).get("done_ids", []))


def save_progress(judge_dir: Path, done_ids: set[int]) -> None:
    p = judge_dir / "progress.json"
    p.write_text(json.dumps({"done_ids": sorted(done_ids)}), encoding="utf-8")


def append_verdict(judge_dir: Path, verdict: dict, allowed_winners: set[str]) -> None:
    assert set(verdict.keys()) >= REQUIRED_KEYS, f"missing keys: {verdict}"
    assert verdict["winner"] in allowed_winners, f"bad winner: {verdict}"
    out_file = judge_dir / "judged_eval_results_v2.jsonl"
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(verdict, ensure_ascii=False) + "\n")


async def judge_one(
    runner, template_str: str, item: dict, rng: random.Random,
    a_key: str, b_key: str,
    max_retries: int = 2,
) -> dict | None:
    if a_key not in item or b_key not in item:
        print(f"  [{runner.name}] id={item.get('id')} SKIP: missing {a_key} or {b_key}")
        return None

    a_first = rng.random() < 0.5
    response_a = item[a_key] if a_first else item[b_key]
    response_b = item[b_key] if a_first else item[a_key]

    prompt = render_prompt(
        template_str,
        category=item["category"],
        prompt=item["prompt"],
        response_a=response_a,
        response_b=response_b,
    )

    parsed = None
    last_err = None
    for attempt in range(max_retries):
        try:
            result = await asyncio.to_thread(runner.call, prompt)
        except Exception as e:
            last_err = e
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        parsed = parse_judge_reply(result["text"])
        if parsed is not None:
            break
        last_err = f"unparseable: {result['text'][:200]}"
        await asyncio.sleep(0.5)

    if parsed is None:
        print(f"  [{runner.name}] id={item['id']} FAILED after {max_retries}: {last_err}")
        return None

    winner = map_ab_to_labels(
        parsed["winner"], winner_label_for(a_key), winner_label_for(b_key), a_first,
    )
    return {
        "id": item["id"],
        "category": item["category"],
        "winner": winner,
        "reason": parsed["reason"],
    }


async def run_judge(
    judge_name: str, items: list[dict], template_str: str,
    out_root: Path, resume: bool, concurrency: int, seed: int,
    a_key: str, b_key: str, allowed_winners: set[str],
    comparison_label: str | None,
) -> None:
    runner = build_judge_runner(judge_name)
    judge_dir = out_root / runner.out_dir_name
    if comparison_label:
        judge_dir = judge_dir / comparison_label
    judge_dir.mkdir(parents=True, exist_ok=True)

    done = load_progress(judge_dir) if resume else set()
    if resume and done:
        print(f"[{judge_name}/{comparison_label or 'default'}] resuming, skipping {len(done)} already-judged ids")
    elif not resume:
        out_f = judge_dir / "judged_eval_results_v2.jsonl"
        if out_f.exists():
            out_f.unlink()

    todo = [it for it in items if it["id"] not in done]
    if not todo:
        print(f"[{judge_name}] nothing to do — all {len(items)} ids already done")
        return

    print(f"[{judge_name}] {len(todo)} items, concurrency={concurrency}, a={a_key} b={b_key}")
    sem = asyncio.Semaphore(concurrency)
    rng = random.Random(seed)
    completed = 0
    failed = 0
    t0 = time.time()
    lock = asyncio.Lock()

    async def worker(item):
        nonlocal completed, failed
        item_rng = random.Random(rng.randint(0, 2**31) + item["id"])
        async with sem:
            verdict = await judge_one(runner, template_str, item, item_rng, a_key, b_key)
        async with lock:
            if verdict is None:
                failed += 1
            else:
                append_verdict(judge_dir, verdict, allowed_winners)
                done.add(item["id"])
                save_progress(judge_dir, done)
                completed += 1
            shown = completed + failed
            if shown % 10 == 0 or shown == len(todo):
                rate = shown / max(time.time() - t0, 0.001)
                print(f"  [{judge_name}] {shown}/{len(todo)}  ok={completed}  fail={failed}  ({rate:.1f}/s)")

    await asyncio.gather(*(worker(it) for it in todo))
    print(f"[{judge_name}] DONE  ok={completed}  fail={failed}  {time.time()-t0:.1f}s")


async def main_async(args):
    if not args.outputs.exists():
        raise SystemExit(f"outputs file not found: {args.outputs}")
    items = load_outputs(args.outputs)
    if args.limit:
        items = items[:args.limit]
    print(f"Loaded {len(items)} eval items from {args.outputs}")

    template_str = load_judge_template()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    allowed_winners = {
        winner_label_for(args.response_a_key),
        winner_label_for(args.response_b_key),
        "tie",
    }

    # Claude (subprocess) → low concurrency. OpenAI/Gemini → higher.
    judges_concurrency = {"openai": 5, "gemini": 5, "claude": 2}

    tasks = []
    for j in args.judges:
        tasks.append(run_judge(
            judge_name=j,
            items=items,
            template_str=template_str,
            out_root=args.out_dir,
            resume=args.resume,
            concurrency=judges_concurrency.get(j, 3),
            seed=args.seed,
            a_key=args.response_a_key,
            b_key=args.response_b_key,
            allowed_winners=allowed_winners,
            comparison_label=args.comparison_label,
        ))
    await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_JUDGES_DIR)
    parser.add_argument(
        "--judges", nargs="+",
        choices=list(JUDGE_DIR_NAMES.keys()),
        default=list(JUDGE_DIR_NAMES.keys()),
    )
    parser.add_argument("--limit", type=int, default=None, help="Only judge first N items (smoke test)")
    parser.add_argument("--resume", action="store_true", help="Skip ids already in progress.json")
    parser.add_argument("--seed", type=int, default=42, help="Seed for A/B position randomization")
    parser.add_argument(
        "--response-a-key", default=DEFAULT_A_KEY,
        help=f"Field name for response A in outputs.jsonl (default: {DEFAULT_A_KEY})",
    )
    parser.add_argument(
        "--response-b-key", default=DEFAULT_B_KEY,
        help=f"Field name for response B in outputs.jsonl (default: {DEFAULT_B_KEY})",
    )
    parser.add_argument(
        "--comparison-label", default=None,
        help="Subdirectory under each judge dir (e.g. 'rag_vs_ft'). "
             "Lets multiple comparisons coexist without clobbering each other.",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
