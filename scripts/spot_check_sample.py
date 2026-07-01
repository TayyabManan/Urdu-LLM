"""
Stratified sampler + self-contained HTML reviewer for the v3 data spot-check.

Pulls ~200 examples across the three generated datasets with GUARANTEED coverage
of every sub-type (so a bug in a rare slice can't hide), oversampling the riskiest
slices (RAG `noise`, reasoning). Writes:

  data/spot_check/sample.jsonl   - the sampled rows (+ sc_* tags), for audit
  data/spot_check/review.html    - offline single-file reviewer (open in a browser)

The reviewer embeds the data (no server / no fetch), renders Urdu in Nastaliq RTL
and Roman in a Latin font, and exports verdicts.jsonl. Feed that to
scripts/spot_check_report.py for the accept/regen decision.

Run:  .venv-eval/bin/python scripts/spot_check_sample.py            (default ~200)
      .venv-eval/bin/python scripts/spot_check_sample.py --total 120 --seed 7
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

GRAMMAR = "data/grammar_pairs/grammar_pairs.jsonl"
SUMMREASON = "data/summ_reason/summ_reason.jsonl"
RAG = "data/rag_triples/rag_triples.jsonl"
OUT_DIR = Path("data/spot_check")


def load(path):
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        r["_idx"] = i
        rows.append(r)
    return rows


def allocate(sizes, budget, floor=2):
    """Give each non-empty stratum `floor` first (guaranteed coverage), then split
    the remainder proportionally to stratum size, never exceeding what's available."""
    strata = list(sizes)
    take = {s: min(floor, sizes[s]) for s in strata}
    remaining = max(0, budget - sum(take.values()))
    cap = {s: sizes[s] - take[s] for s in strata}
    wsum = sum(sizes.values())
    if remaining > 0 and wsum > 0:
        ideal = {s: remaining * sizes[s] / wsum for s in strata}
        for s in strata:
            take[s] += min(int(ideal[s]), cap[s])
        leftover = budget - sum(take.values())
        order = sorted(strata, key=lambda s: ideal[s] - int(ideal[s]), reverse=True)
        i = 0
        guard = 0
        while leftover > 0 and any(take[s] < sizes[s] for s in strata) and guard < 100000:
            s = order[i % len(order)]
            if take[s] < sizes[s]:
                take[s] += 1
                leftover -= 1
            i += 1
            guard += 1
    return take


def sample_group(rows, stratum_fn, budget, floor, rng, dataset, group):
    buckets = {}
    for r in rows:
        buckets.setdefault(stratum_fn(r), []).append(r)
    sizes = {k: len(v) for k, v in buckets.items()}
    plan = allocate(sizes, budget, floor=floor)
    picked = []
    for stratum, n in plan.items():
        chosen = rng.sample(buckets[stratum], min(n, len(buckets[stratum])))
        for r in chosen:
            rec = dict(r)
            rec["sc_id"] = f"{dataset}#{r['_idx']}"
            rec["sc_dataset"] = dataset
            rec["sc_group"] = group
            rec["sc_stratum"] = stratum
            picked.append(rec)
    return picked, plan


def build(total, seed):
    rng = random.Random(seed)
    grammar = load(GRAMMAR)
    sr = load(SUMMREASON)
    rag = load(RAG)
    reason = [r for r in sr if r.get("kind") == "reasoning"]
    summ = [r for r in sr if r.get("kind") == "summarization"]

    # Per-group budgets scaled to `total` (defaults sum to 200). Oversample the
    # v3-critical slices: RAG noise and reasoning.
    scale = total / 200.0
    b = lambda n: max(1, round(n * scale))
    sample = []
    summary = []

    def add(rows, fn, budget, floor, dataset, group):
        picked, plan = sample_group(rows, fn, budget, floor, rng, dataset, group)
        sample.extend(picked)
        summary.append((group, len(picked), plan))

    add(grammar, lambda r: f"{r['error_type']}/{r['script']}", b(50), 2, "grammar", "grammar")
    add(summ, lambda r: "summarization", b(35), b(35), "summ_reason", "summarization")
    add(reason, lambda r: f"{r['template_name']}/{'roman' if r['is_roman'] else 'urdu'}",
        b(35), 2, "summ_reason", "reasoning")
    add([r for r in rag if r["rag_kind"] == "single"], lambda r: "single", b(24), b(24), "rag", "rag_single")
    add([r for r in rag if r["rag_kind"] == "multi"], lambda r: "multi", b(26), b(26), "rag", "rag_multi")
    add([r for r in rag if r["rag_kind"] == "noise"], lambda r: "noise", b(30), b(30), "rag", "rag_noise")

    rng.shuffle(sample)  # interleave datasets so the reviewer isn't doing 50 of one kind in a row

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "sample.jsonl", "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    html = HTML_TEMPLATE.replace(
        "/*__DATA__*/",
        json.dumps([slim(r) for r in sample], ensure_ascii=False),
    )
    (OUT_DIR / "review.html").write_text(html, encoding="utf-8")

    print(f"Sampled {len(sample)} examples (seed={seed}) -> {OUT_DIR}/sample.jsonl")
    print(f"Reviewer -> {OUT_DIR}/review.html  (open in a browser)\n")
    for group, n, plan in summary:
        print(f"  {group:14s} {n:3d}   strata: " +
              ", ".join(f"{k}={v}" for k, v in sorted(plan.items())))


def slim(r):
    """Only the fields the reviewer needs (keeps the embedded blob small)."""
    return {
        "sc_id": r["sc_id"],
        "group": r["sc_group"],
        "stratum": r["sc_stratum"],
        "system": r.get("system", ""),
        "instruction": r.get("instruction", ""),
        "output": r.get("output", ""),
        "error_type": r.get("error_type", ""),
        "template_name": r.get("template_name", ""),
        "final_answer": r.get("final_answer", ""),
        "gold_title": r.get("gold_title") or "",
        "n_context_chunks": r.get("n_context_chunks", ""),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Urdu v3 — Data Spot-Check</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root{--bg:#0f1115;--card:#181b22;--mut:#8b93a3;--bd:#2a2f3a;--fg:#e8eaf0;
        --pass:#2ea043;--fail:#e5534b;--uns:#d9a40a;--accent:#4c8bf5;--hint:#3a3320;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
  header{position:sticky;top:0;background:#12141a;border-bottom:1px solid var(--bd);
         padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:5;}
  header .sp{flex:1}
  .counts b{color:var(--pass)} .counts i{color:var(--fail);font-style:normal} .counts u{color:var(--uns);text-decoration:none}
  button{background:#222734;color:var(--fg);border:1px solid var(--bd);border-radius:8px;
         padding:8px 12px;font-size:14px;cursor:pointer}
  button:hover{border-color:var(--accent)}
  .wrap{max-width:920px;margin:18px auto;padding:0 16px}
  .card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:18px}
  .badges{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;font-size:12px}
  .b{padding:3px 9px;border-radius:999px;background:#222734;color:var(--mut);border:1px solid var(--bd)}
  .b.g{color:#cdd6e6}
  .hint{background:var(--hint);border:1px solid #5a4d22;color:#f0e6c0;border-radius:8px;
        padding:9px 12px;font-size:13.5px;margin:10px 0}
  .lbl{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:14px 0 4px}
  .box{background:#0e1016;border:1px solid var(--bd);border-radius:8px;padding:12px 14px;white-space:pre-wrap;word-wrap:break-word}
  .ctx{max-height:230px;overflow:auto}
  .urdu{font-family:'Noto Nastaliq Urdu',serif;direction:rtl;text-align:right;line-height:2.3;font-size:1.18rem}
  .latin{font-family:'JetBrains Mono',ui-monospace,Menlo,Consolas,monospace;direction:ltr;line-height:1.65;font-size:.98rem}
  .q{border-left:3px solid var(--accent);padding-left:10px}
  .verdicts{display:flex;gap:10px;margin:16px 0 8px}
  .verdicts button{flex:1;font-weight:600}
  .v-pass.on{background:var(--pass);border-color:var(--pass);color:#fff}
  .v-fail.on{background:var(--fail);border-color:var(--fail);color:#fff}
  .v-uns.on{background:var(--uns);border-color:var(--uns);color:#1a1a1a}
  textarea{width:100%;min-height:52px;background:#0e1016;color:var(--fg);border:1px solid var(--bd);
           border-radius:8px;padding:8px 10px;font-size:14px;resize:vertical}
  .nav{display:flex;justify-content:space-between;margin-top:14px}
  .foot{color:var(--mut);font-size:12px;text-align:center;margin:14px auto 40px}
  kbd{background:#222734;border:1px solid var(--bd);border-radius:5px;padding:1px 6px;font-size:11px}
  table{width:100%;border-collapse:collapse;margin-top:8px;font-size:13px}
  td,th{border:1px solid var(--bd);padding:6px 9px;text-align:left}
  .low{color:var(--fail);font-weight:600}
  .ok{color:var(--pass)}
  dialog{background:var(--card);color:var(--fg);border:1px solid var(--bd);border-radius:12px;max-width:680px;width:92%}
</style>
</head>
<body>
<header>
  <strong>Urdu v3 spot-check</strong>
  <span class="b g" id="pos">–</span>
  <span class="counts" id="counts"></span>
  <span class="sp"></span>
  <button id="summaryBtn">Summary</button>
  <button id="exportBtn">⬇ Export verdicts</button>
</header>
<div class="wrap">
  <div class="card" id="card"></div>
  <div class="foot">
    <kbd>P</kbd> pass · <kbd>F</kbd> fail · <kbd>U</kbd> unsure · <kbd>←</kbd>/<kbd>→</kbd> prev/next · <kbd>Enter</kbd> next ·
    progress auto-saves to this browser
  </div>
</div>
<dialog id="dlg"><div id="dlgBody"></div><div style="text-align:right;margin-top:12px"><button onclick="document.getElementById('dlg').close()">Close</button></div></dialog>
<script>
const DATA = /*__DATA__*/;
const LS = "urdu_spotcheck_v3";
let verd = JSON.parse(localStorage.getItem(LS) || "{}");
let cur = 0;

const HINTS = {
  grammar: "Check: exactly ONE real error of the labelled type in the prompt, and the gold (output) is natural, correct Urdu (gold should be the trusted seed, not an AI rewrite).",
  summarization: "Check: summary is faithful to the passage (no invented facts), is ~3 sentences, and reads as natural Urdu. Watch for leaked Wikipedia boilerplate in the passage.",
  reasoning: "Check: the narration matches the arithmetic, the final answer is correct and in digits, and the word problem is sensible.",
  rag_single: "Check: the answer is grounded in the single context chunk and correct (not invented).",
  rag_multi: "Check: the answer uses the GOLD chunk and correct — it must ignore the 2 distractor chunks.",
  rag_noise: "CRITICAL: the context is irrelevant on purpose. The answer must DEFER (‘not in context → general knowledge…’), NOT fabricate an answer from the unrelated chunks.",
};

function scriptClass(t){
  if(!t) return 'latin';
  const ar=(t.match(/[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]/g)||[]).length;
  const la=(t.match(/[A-Za-z]/g)||[]).length;
  return ar>=la ? 'urdu':'latin';
}
function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function blk(text,cls){const sc=scriptClass(text);return `<div class="box ${cls||''} ${sc}">${esc(text)}</div>`;}

function render(){
  const d = DATA[cur];
  const v = verd[d.sc_id] || {};
  let body = `<div class="badges">
      <span class="b g">${d.group}</span>
      <span class="b">${esc(d.stratum)}</span>
      ${d.error_type?`<span class="b">error: ${esc(d.error_type)}</span>`:''}
      ${d.template_name?`<span class="b">tmpl: ${esc(d.template_name)}</span>`:''}
      ${d.final_answer?`<span class="b">answer: ${esc(d.final_answer)}</span>`:''}
      ${d.gold_title?`<span class="b">gold: ${esc(d.gold_title)}</span>`:''}
      <span class="b">${esc(d.sc_id)}</span>
    </div>
    <div class="hint">${HINTS[d.group]||''}</div>`;

  if(d.group.startsWith("rag_")){
    if(d.system){ body += `<div class="lbl">system</div>${blk(d.system,'')}`; }
    const parts = d.instruction.split("سوال:");
    const ctx = parts[0].trim();
    const q = parts.length>1 ? parts.slice(1).join("سوال:").trim() : "";
    body += `<div class="lbl">context (${d.n_context_chunks} chunk${d.n_context_chunks==1?'':'s'})</div>${blk(ctx,'ctx')}`;
    if(q) body += `<div class="lbl">question</div>${blk(q,'q')}`;
    body += `<div class="lbl">model answer (gold target)</div>${blk(d.output,'')}`;
  } else {
    body += `<div class="lbl">prompt / instruction</div>${blk(d.instruction,'')}`;
    body += `<div class="lbl">gold output (training target)</div>${blk(d.output,'')}`;
  }

  body += `<div class="verdicts">
      <button class="v-pass ${v.verdict==='pass'?'on':''}" onclick="setV('pass')">✓ Pass (p)</button>
      <button class="v-fail ${v.verdict==='fail'?'on':''}" onclick="setV('fail')">✗ Fail (f)</button>
      <button class="v-uns ${v.verdict==='unsure'?'on':''}" onclick="setV('unsure')">? Unsure (u)</button>
    </div>
    <textarea id="note" placeholder="note (optional) — what's wrong, if anything">${esc(v.note||'')}</textarea>
    <div class="nav"><button onclick="go(-1)">← Prev</button><button onclick="go(1)">Next →</button></div>`;

  document.getElementById('card').innerHTML = body;
  document.getElementById('note').addEventListener('input', e=>{
    const id=DATA[cur].sc_id; verd[id]=Object.assign({},verd[id],{note:e.target.value}); save();
  });
  document.getElementById('pos').textContent = `${cur+1} / ${DATA.length}`;
  updateCounts();
}
function updateCounts(){
  let p=0,f=0,u=0,done=0;
  for(const d of DATA){const v=verd[d.sc_id];if(v&&v.verdict){done++;if(v.verdict=='pass')p++;else if(v.verdict=='fail')f++;else u++;}}
  document.getElementById('counts').innerHTML =
    `reviewed ${done}/${DATA.length} · <b>✓${p}</b> <i>✗${f}</i> <u>?${u}</u>`;
}
function setV(val){const id=DATA[cur].sc_id;verd[id]=Object.assign({},verd[id],{verdict:val});save();render();setTimeout(()=>go(1),120);}
function save(){localStorage.setItem(LS, JSON.stringify(verd));}
function go(d){cur=Math.max(0,Math.min(DATA.length-1,cur+d));render();window.scrollTo(0,0);}

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'){ if(e.key==='Escape')e.target.blur(); return; }
  if(e.key==='p')setV('pass'); else if(e.key==='f')setV('fail'); else if(e.key==='u')setV('unsure');
  else if(e.key==='ArrowRight'||e.key==='Enter')go(1); else if(e.key==='ArrowLeft')go(-1);
});

document.getElementById('exportBtn').onclick=()=>{
  const lines=DATA.map(d=>{const v=verd[d.sc_id]||{};return JSON.stringify({
    sc_id:d.sc_id,dataset:d.sc_id.split('#')[0],group:d.group,stratum:d.stratum,
    verdict:v.verdict||null,note:v.note||""});}).join("\n");
  const blob=new Blob([lines],{type:"application/x-ndjson"});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download="verdicts.jsonl";a.click();
};
document.getElementById('summaryBtn').onclick=()=>{
  const g={};
  for(const d of DATA){const v=verd[d.sc_id]||{};const k=d.group;g[k]=g[k]||{p:0,f:0,u:0,n:0};
    g[k].n++; if(v.verdict=='pass')g[k].p++; else if(v.verdict=='fail')g[k].f++; else if(v.verdict=='unsure')g[k].u++;}
  let rows="";
  for(const k of Object.keys(g)){const s=g[k];const rev=s.p+s.f+s.u;
    const rate=rev? (100*s.p/rev):0; const cls=rev&&rate<80?'low':'ok';
    rows+=`<tr><td>${k}</td><td>${s.n}</td><td>${rev}</td><td class="${cls}">${rev?rate.toFixed(0)+'%':'–'}</td><td>${s.f}</td><td>${s.u}</td></tr>`;}
  document.getElementById('dlgBody').innerHTML=
    `<h3>Per-group pass rate</h3><table><tr><th>group</th><th>sampled</th><th>reviewed</th><th>pass%</th><th>fail</th><th>unsure</th></tr>${rows}</table>
     <p style="color:var(--mut);font-size:13px;margin-top:12px">Green-light rule of thumb: overall ≥90% pass and no group below 80%. Red groups → fix that generator and regenerate just that slice before training.</p>`;
  document.getElementById('dlg').showModal();
};
render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=200, help="approx total examples to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.total, args.seed)


if __name__ == "__main__":
    main()
