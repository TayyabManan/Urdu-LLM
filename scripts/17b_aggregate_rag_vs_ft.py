"""Aggregate RAG-vs-FT judge verdicts (mirror of 17_aggregate_judges.py but
for the rag_vs_ft comparison label).

Reads data/eval/v2/judges/*/rag_vs_ft/judged_eval_results_v2.jsonl and
prints + plots RAG win rate vs plain-FT, per-judge per-category.

Usage:
  python scripts/17b_aggregate_rag_vs_ft.py
  python scripts/17b_aggregate_rag_vs_ft.py --out rag_vs_ft.png
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
JUDGES_DIR = ROOT / "data" / "eval" / "v2" / "judges"

CATEGORIES = [
    "translation", "code_mixed", "qa", "creative",
    "code_explanation", "reasoning", "summarization", "grammar",
]
CATEGORY_LABELS = {
    "translation": "Translation", "code_mixed": "Code\nMixed", "qa": "QA",
    "creative": "Creative", "code_explanation": "Code\nExplanation",
    "reasoning": "Reasoning", "summarization": "Summarization", "grammar": "Grammar",
}


def load_judge(jdir: Path, comparison: str):
    f = jdir / comparison / "judged_eval_results_v2.jsonl"
    if not f.exists():
        return None
    recs = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def stats(recs, win_key: str):
    by_cat = defaultdict(Counter)
    for r in recs:
        by_cat[r["category"]][r["winner"]] += 1
    out = {}
    for cat, c in by_cat.items():
        n = c["finetuned"] + c[win_key] + c["tie"]
        out[cat] = {
            "rag_pct": 100 * c[win_key] / n if n else 0.0,
            "ft_pct": 100 * c["finetuned"] / n if n else 0.0,
            "tie_pct": 100 * c["tie"] / n if n else 0.0,
            "rag": c[win_key], "ft": c["finetuned"], "tie": c["tie"], "n": n,
        }
    n_total = sum(out[c]["n"] for c in out)
    rag_total = sum(out[c]["rag"] for c in out)
    ft_total = sum(out[c]["ft"] for c in out)
    tie_total = sum(out[c]["tie"] for c in out)
    out["overall"] = {
        "rag_pct": 100 * rag_total / n_total if n_total else 0.0,
        "ft_pct": 100 * ft_total / n_total if n_total else 0.0,
        "tie_pct": 100 * tie_total / n_total if n_total else 0.0,
        "rag": rag_total, "ft": ft_total, "tie": tie_total, "n": n_total,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges-dir", default=str(JUDGES_DIR),
                        help="directory of per-judge verdict folders (default: v2)")
    parser.add_argument("--out-dir", default=None,
                        help="directory for the output PNG (default: the judges-dir parent)")
    parser.add_argument("--comparison", default="rag_vs_ft",
                        help="Comparison label subdir (e.g. rag_vs_ft, rag_base_vs_ft)")
    parser.add_argument("--win-key", default="rag",
                        help="Winner label for the 'B' side (rag or rag_base)")
    parser.add_argument("--out", default=None,
                        help="Output filename (defaults to <comparison>_aggregate.png)")
    args = parser.parse_args()
    out_name = args.out or f"{args.comparison}_aggregate.png"
    judges_dir = Path(args.judges_dir)
    out_base = Path(args.out_dir) if args.out_dir else judges_dir.parent

    judges: dict[str, dict] = {}
    for jdir in sorted(judges_dir.iterdir()):
        if not jdir.is_dir():
            continue
        recs = load_judge(jdir, args.comparison)
        if not recs:
            continue
        judges[jdir.name] = stats(recs, args.win_key)
    if not judges:
        raise SystemExit(f"No {args.comparison}/ subdirs found under {judges_dir}")

    print(f"Judges loaded: {list(judges)}")
    print()

    header = f"{'Category':<20}"
    for j in judges:
        header += f"{j[:22]:<24}"
    header += f"{'Median':<10}"
    print(header)
    print("-" * len(header))

    rag_pcts_overall = []
    for cat in CATEGORIES + ["overall"]:
        rag_pcts = [judges[j][cat]["rag_pct"] for j in judges if cat in judges[j]]
        if not rag_pcts:
            continue
        median = np.median(rag_pcts)
        line = f"{cat:<20}"
        for j in judges:
            if cat in judges[j]:
                s = judges[j][cat]
                line += f"  RAG {s['rag_pct']:>4.1f}% ({s['rag']}/{s['n']})   "
            else:
                line += f"  {'-':<24}"
        line += f"{median:>5.1f}%"
        print(line)
        if cat == "overall":
            rag_pcts_overall = rag_pcts

    # Plot per-category RAG win rate (median dot, judge dots, 50% baseline)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor("white")

    cats_sorted = sorted(
        CATEGORIES,
        key=lambda c: -np.median([judges[j][c]["rag_pct"] for j in judges if c in judges[j]] or [0]),
    )
    x = np.arange(len(cats_sorted))
    judge_names = list(judges)
    colors = ["#2563eb", "#10b981", "#f59e0b", "#a855f7"]

    for i, cat in enumerate(cats_sorted):
        pcts = [judges[j][cat]["rag_pct"] for j in judges if cat in judges[j]]
        if not pcts:
            continue
        if len(pcts) > 1:
            ax.vlines(i, min(pcts), max(pcts), color="#cbd5e1", linewidth=4, zorder=1)

    for k, jname in enumerate(judge_names):
        pcts = [judges[jname][c]["rag_pct"] if c in judges[jname] else None for c in cats_sorted]
        xs = [i for i, p in enumerate(pcts) if p is not None]
        ys = [p for p in pcts if p is not None]
        ax.scatter(xs, ys, s=70, color=colors[k % len(colors)],
                   label=jname, zorder=3, edgecolor="white", linewidth=1.2)

    medians = []
    for c in cats_sorted:
        pcts = [judges[j][c]["rag_pct"] for j in judges if c in judges[j]]
        medians.append(np.median(pcts) if pcts else 0)
    ax.scatter(x, medians, s=180, marker="_", color="#1f2937", linewidth=2.5,
               zorder=4, label="median")

    ax.axhline(50, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.6, label="50% (parity)")

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats_sorted], fontsize=10)
    ax.set_ylabel("RAG-FT win rate vs plain-FT (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title(
        f"Urdu {args.comparison} (hybrid+rerank, top_k=3) — {len(judges)} judge(s)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    if rag_pcts_overall:
        med = float(np.median(rag_pcts_overall))
        fig.text(0.5, 0.01,
                 f"Overall RAG win rate: median {med:.1f}%  ·  range {min(rag_pcts_overall):.1f}–{max(rag_pcts_overall):.1f}%  ·  100 prompts × {len(judges)} judge(s)",
                 ha="center", fontsize=9, color="#374151", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = out_base / out_name
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
