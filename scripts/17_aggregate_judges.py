"""
Aggregate v2 judge verdicts and produce a comparison chart.
Reads judge folders under data/eval/v2/judges/ and produces:
- Per-judge per-category FT win rates (printed)
- Aggregate chart showing range of judge verdicts per category
  (median tick, judge dots, v1 baseline, 60% target line)

Usage:
  python scripts/18_aggregate_judges.py                          # all judges -> aggregate.png
  python scripts/18_aggregate_judges.py --exclude claude-code-opus-4.7   # excludes one
  python scripts/18_aggregate_judges.py --out custom_name.png    # custom output filename
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
JUDGES_DIR = ROOT / "data" / "eval" / "v2" / "judges"

parser = argparse.ArgumentParser()
parser.add_argument("--exclude", nargs="*", default=[], help="judge folder names to exclude")
parser.add_argument("--out", default=None, help="output filename (placed in data/eval/v2/)")
args = parser.parse_args()

if args.out:
    out_name = args.out
elif args.exclude:
    out_name = f"aggregate_excl_{'_'.join(args.exclude)}.png"
else:
    out_name = "aggregate.png"
OUT_PNG = ROOT / "data" / "eval" / "v2" / out_name

CATEGORIES = ["translation", "code_mixed", "qa", "creative",
              "code_explanation", "reasoning", "summarization", "grammar"]
CATEGORY_LABELS = {
    "translation": "Translation",
    "code_mixed": "Code\nMixed",
    "qa": "QA",
    "creative": "Creative",
    "code_explanation": "Code\nExplanation",
    "reasoning": "Reasoning",
    "summarization": "Summarization",
    "grammar": "Grammar",
}
V1 = {"translation": 80.0, "code_mixed": 0.0, "qa": 61.9, "creative": 54.5,
      "code_explanation": 10.0, "reasoning": 43.8, "summarization": 72.7, "grammar": 40.0,
      "overall": 51.5}

def load_judge(jdir):
    f = jdir / "judged_eval_results_v2.jsonl"
    if not f.exists(): return None
    recs = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs

def stats(recs):
    by_cat = defaultdict(lambda: Counter())
    for r in recs:
        by_cat[r["category"]][r["winner"]] += 1
    out = {}
    for cat, c in by_cat.items():
        total = c["finetuned"] + c["base"] + c["tie"]
        out[cat] = {"ft_pct": 100 * c["finetuned"] / total if total else 0,
                    "ft": c["finetuned"], "base": c["base"], "tie": c["tie"], "total": total}
    overall_ft = sum(c["finetuned"] for c in by_cat.values())
    overall_total = sum(c["finetuned"]+c["base"]+c["tie"] for c in by_cat.values())
    out["overall"] = {"ft_pct": 100*overall_ft/overall_total, "ft": overall_ft,
                      "base": sum(c["base"] for c in by_cat.values()),
                      "tie": sum(c["tie"] for c in by_cat.values()), "total": overall_total}
    return out

judges = {}
for jdir in sorted(JUDGES_DIR.iterdir()):
    if not jdir.is_dir(): continue
    if jdir.name in args.exclude: continue
    recs = load_judge(jdir)
    if recs is None: continue
    judges[jdir.name] = stats(recs)

print(f"Including judges: {list(judges)}")
if args.exclude: print(f"Excluded: {args.exclude}")

print(f"\n{'Category':<20}", end="")
for jname in judges: print(f"{jname:<28}", end="")
print(f"{'Median':<10}{'v1':<8}{'Δ vs v1':<10}")
print("-" * (20 + 28*len(judges) + 28))
for cat in CATEGORIES + ["overall"]:
    pcts = [judges[j][cat]["ft_pct"] for j in judges if cat in judges[j]]
    median = np.median(pcts)
    v1 = V1[cat]
    delta = median - v1
    print(f"{cat:<20}", end="")
    for j in judges:
        s = judges[j][cat]
        print(f"  {s['ft_pct']:>5.1f}% ({s['ft']}/{s['total']})         ", end="")
    print(f"{median:>5.1f}%   {v1:>4.1f}%   {delta:+.1f}")

fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
fig.patch.set_facecolor("white")

cats_sorted = sorted(CATEGORIES, key=lambda c: -np.median([judges[j][c]["ft_pct"] for j in judges]))
x = np.arange(len(cats_sorted))

judge_colors = {"claude-code-opus-4.7": "#2563eb",
                "claude-desktop-opus-4.7": "#60a5fa",
                "gemini-3.1-pro": "#10b981",
                "gpt-5.3-thinking": "#f59e0b"}
judge_short = {"claude-code-opus-4.7": "Claude Code (Opus 4.7)",
               "claude-desktop-opus-4.7": "Claude Desktop (Opus 4.7)",
               "gemini-3.1-pro": "Gemini 3.1 Pro",
               "gpt-5.3-thinking": "GPT 5.3 Thinking"}

for i, cat in enumerate(cats_sorted):
    pcts = [judges[j][cat]["ft_pct"] for j in judges]
    ax.vlines(i, min(pcts), max(pcts), color="#cbd5e1", linewidth=4, zorder=1)

for jname in judges:
    pcts = [judges[jname][c]["ft_pct"] for c in cats_sorted]
    ax.scatter(x, pcts, s=70, color=judge_colors.get(jname, "#666"),
               label=judge_short.get(jname, jname), zorder=3,
               edgecolor="white", linewidth=1.2)

medians = [np.median([judges[j][c]["ft_pct"] for j in judges]) for c in cats_sorted]
ax.scatter(x, medians, s=180, marker="_", color="#1f2937", linewidth=2.5, zorder=4,
           label="median")

ax.axhline(60, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.6, label="60% target")
ax.axhline(50, color="#9ca3af", linestyle=":", linewidth=1, alpha=0.5)

for i, cat in enumerate(cats_sorted):
    v1_val = V1[cat]
    ax.scatter(i, v1_val, s=60, marker="x", color="#6b7280", zorder=2, alpha=0.8)

ax.scatter([], [], s=60, marker="x", color="#6b7280", alpha=0.8, label="v1 baseline")

ax.set_xticks(x)
ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats_sorted], fontsize=10)
ax.set_ylabel("Win rate vs base model (%)", fontsize=11)
ax.set_ylim(-5, 110)
ax.set_yticks(range(0, 101, 20))
ax.set_title(f"Urdu LLM v2 — {len(judges)} Judges Compared by Category",
             fontsize=14, fontweight="bold", pad=14)
ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.4)

overall_pcts = [judges[j]["overall"]["ft_pct"] for j in judges]
overall_med = np.median(overall_pcts)
fig.text(0.5, 0.01,
         f"Overall: median {overall_med:.1f}%  ·  range {min(overall_pcts):.1f}–{max(overall_pcts):.1f}%  ·  v1 baseline 51.5%  ·  100 prompts × {len(judges)} LLM judges",
         ha="center", fontsize=9, color="#374151", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
print(f"\nsaved: {OUT_PNG}")
