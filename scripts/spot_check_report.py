"""
Tally spot-check verdicts.jsonl into an accept / regenerate decision.

Reads the verdicts exported by data/spot_check/review.html and joins them with
data/spot_check/sample.jsonl. Prints pass rates per group and per stratum and
applies the green-light rule:

  GREEN  : overall pass >= --overall (default 90%) AND every group >= --group (default 80%)
  REGEN  : otherwise — lists the groups/strata to fix + regenerate before training.

Run:  .venv-eval/bin/python scripts/spot_check_report.py
      .venv-eval/bin/python scripts/spot_check_report.py --verdicts data/spot_check/verdicts.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DIR = Path("data/spot_check")


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def rate(d):
    rev = d["pass"] + d["fail"] + d["unsure"]
    return (100 * d["pass"] / rev) if rev else None


def tally(verdicts, key):
    out = defaultdict(lambda: {"n": 0, "pass": 0, "fail": 0, "unsure": 0, "fails": []})
    for v in verdicts:
        b = out[v[key]]
        b["n"] += 1
        ver = v.get("verdict")
        if ver in ("pass", "fail", "unsure"):
            b[ver] += 1
        if ver == "fail":
            b["fails"].append((v["sc_id"], v.get("note", "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default=str(DIR / "verdicts.jsonl"))
    ap.add_argument("--overall", type=float, default=90.0)
    ap.add_argument("--group", type=float, default=80.0)
    args = ap.parse_args()

    vpath = Path(args.verdicts)
    if not vpath.exists():
        sys.exit(f"No verdicts file at {vpath}. Export it from review.html first.")
    verdicts = load_jsonl(vpath)

    total = len(verdicts)
    reviewed = sum(1 for v in verdicts if v.get("verdict") in ("pass", "fail", "unsure"))
    npass = sum(1 for v in verdicts if v.get("verdict") == "pass")
    nfail = sum(1 for v in verdicts if v.get("verdict") == "fail")
    nuns = sum(1 for v in verdicts if v.get("verdict") == "unsure")
    overall = (100 * npass / reviewed) if reviewed else 0.0

    print(f"\nSpot-check report  ({vpath})")
    print(f"  sampled {total} | reviewed {reviewed} | ✓ {npass}  ✗ {nfail}  ? {nuns}")
    print(f"  overall pass rate: {overall:.1f}%  (on reviewed)\n")
    if reviewed < total:
        print(f"  ⚠ {total - reviewed} examples not yet reviewed.\n")

    by_group = tally(verdicts, "group")
    print("Per group:")
    print(f"  {'group':14s} {'n':>3s} {'rev':>4s} {'pass%':>6s} {'fail':>5s} {'uns':>4s}")
    failing_groups = []
    for g in sorted(by_group):
        d = by_group[g]
        r = rate(d)
        flag = ""
        if r is not None and r < args.group:
            flag = "  <-- below threshold"
            failing_groups.append(g)
        rs = f"{r:.0f}%" if r is not None else "  –"
        rev = d["pass"] + d["fail"] + d["unsure"]
        print(f"  {g:14s} {d['n']:3d} {rev:4d} {rs:>6s} {d['fail']:5d} {d['unsure']:4d}{flag}")

    # stratum detail only for failing groups
    if failing_groups:
        by_str = tally(verdicts, "stratum")
        print("\nFailing-group strata detail:")
        for v in verdicts:
            pass
        # show strata that belong to failing groups, with their fails
        group_of = {v["stratum"]: v["group"] for v in verdicts}
        for s in sorted(by_str):
            if group_of.get(s) in failing_groups:
                d = by_str[s]
                r = rate(d)
                rs = f"{r:.0f}%" if r is not None else "–"
                print(f"  [{group_of[s]}] {s:24s} pass {rs:>5s}  fails={d['fail']}")

    # list a few failure notes to act on
    fails = [(v["group"], v["sc_id"], v.get("note", "")) for v in verdicts if v.get("verdict") == "fail"]
    if fails:
        print(f"\nFailures ({len(fails)}):")
        for g, sid, note in fails[:40]:
            print(f"  [{g}] {sid}  {note}")
        if len(fails) > 40:
            print(f"  ... +{len(fails) - 40} more")

    print("\n" + "=" * 56)
    green = reviewed > 0 and overall >= args.overall and not failing_groups
    if green:
        print(f"DECISION: GREEN — overall {overall:.0f}% ≥ {args.overall:.0f}% and every group ≥ "
              f"{args.group:.0f}%.\n  Proceed to scripts/26 (combine v3) → 27 (train).")
    else:
        print("DECISION: REGEN — do NOT train yet.")
        if reviewed < total:
            print(f"  • finish reviewing ({total - reviewed} left) for a reliable read.")
        if overall < args.overall and reviewed:
            print(f"  • overall {overall:.0f}% < {args.overall:.0f}% target.")
        for g in failing_groups:
            print(f"  • fix the generator for '{g}' and regenerate that slice, then re-spot-check it.")
    print("=" * 56)


if __name__ == "__main__":
    main()
