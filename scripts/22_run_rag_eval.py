"""Generate RAG responses on the 100-prompt eval set.

Loads data/eval/prompts.jsonl, runs each through the RAG pipeline, writes
data/eval/v2/rag_outputs.jsonl with the same schema as outputs.jsonl plus a
`rag_response` field and `retrieved_titles`.

After this, judge RAG-FT vs plain-FT via:
    python scripts/18_run_judges.py \\
        --outputs data/eval/v2/rag_outputs.jsonl \\
        --response-a-key finetuned_response \\
        --response-b-key rag_response \\
        --comparison-label rag_vs_ft \\
        --judges claude openai

Usage:
    export FT_ENDPOINT_URL="https://...modal.run"
    python scripts/22_run_rag_eval.py                  # all 100
    python scripts/22_run_rag_eval.py --limit 5        # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.rag.inference import build_rag_pipeline, query_rag  # noqa: E402

DEFAULT_PROMPTS = ROOT / "data" / "eval" / "prompts.jsonl"
DEFAULT_BASELINE = ROOT / "data" / "eval" / "v2" / "outputs.jsonl"
DEFAULT_OUT = ROOT / "data" / "eval" / "v2" / "rag_outputs.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                        help="Existing outputs.jsonl with base_response + finetuned_response")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--resume", action="store_true",
                        help="Skip ids already in --out")
    args = parser.parse_args()

    if not args.prompts.exists():
        raise SystemExit(f"prompts file not found: {args.prompts}")
    if not args.baseline.exists():
        raise SystemExit(f"baseline file not found: {args.baseline}")

    prompts = load_jsonl(args.prompts)
    baseline_by_id = {row["id"]: row for row in load_jsonl(args.baseline)}
    if args.limit:
        prompts = prompts[:args.limit]
    print(f"Loaded {len(prompts)} prompts, {len(baseline_by_id)} baseline rows")

    done_ids = set()
    if args.resume and args.out.exists():
        done_ids = {row["id"] for row in load_jsonl(args.out)}
        print(f"Resume: skipping {len(done_ids)} already-done ids")
    elif args.out.exists() and not args.resume:
        args.out.unlink()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building RAG pipeline (top_k={args.top_k})...")
    pipeline = build_rag_pipeline(
        top_k=args.top_k,
        embed_device=args.embed_device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    t0 = time.time()
    todo = [p for p in prompts if p["id"] not in done_ids]
    print(f"Generating {len(todo)} RAG responses...")

    with args.out.open("a", encoding="utf-8") as fout:
        for i, p in enumerate(todo, start=1):
            pid = p["id"]
            try:
                result = query_rag(pipeline, p["prompt"])
            except Exception as e:
                print(f"  [id={pid}] FAILED: {e!r}")
                continue
            baseline = baseline_by_id.get(pid, {})
            row = {
                "id": pid,
                "category": p["category"],
                "language": p["language"],
                "prompt": p["prompt"],
                "base_response": baseline.get("base_response", ""),
                "finetuned_response": baseline.get("finetuned_response", ""),
                "rag_response": result["answer"],
                "retrieved_titles": [s.get("title") for s in result["sources"]],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 5 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 0.001)
                print(f"  {i}/{len(todo)}  ({rate:.2f}/s)")

    print(f"\nDONE: {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
