# Urdu LLM — Qwen 2.5 7B Fine-Tune

A 30-day public build of a Qwen 2.5 7B Instruct fine-tune specialised on Urdu
(Urdu script + Roman Urdu + code-mixed), shipped openly with the full pipeline:
data prep → QLoRA training → multi-judge evaluation → optional RAG layer.

🤗 **Model:** [TayyabManan/qwen2.5-7b-urdu-v2](https://huggingface.co/TayyabManan/qwen2.5-7b-urdu-v2)
🎮 **Live demo:** [Space — TayyabManan/urdu-llm-chat](https://huggingface.co/spaces/TayyabManan/urdu-llm-chat)
📋 **Original roadmap:** [`urdu-llm-roadmap.md`](./urdu-llm-roadmap.md)

---

## Headline

**66% median pairwise win rate vs base Qwen 2.5 7B Instruct across 3 independent
LLM judges** on a 100-prompt hand-curated Urdu evaluation set. Range 48–67%.

| Judge | Win rate vs base |
|---|---|
| Claude Desktop (Opus 4.7) | 67.0% |
| GPT 5.3 Thinking | 66.0% |
| Gemini 3.1 Pro | 48.0% |
| **Median (3 judges)** | **66.0%** |

Plus a 4th automated judge via Claude Code CLI subprocess (excluded from
median due to 79% same-model agreement with Claude Desktop): 65.0%.

### Per-category, median across 3 judges

| Category | Win rate | Note |
|---|---|---|
| Creative writing | 91% | strong gain |
| Code-mixed (Urdu/English) | 80% | v1 was 0% |
| Translation (UR↔EN) | 80% | |
| Question Answering | 71% | |
| Code explanation in Urdu | 60% | |
| Summarisation | 46% | **regressed** from v1's 73% |
| Reasoning | 31% | regressed (44%→31%) |
| Grammar correction | 36% | regressed (40%→36%) |

The regressions are the v3 work-list.

---

## Quickstart — try the model

**Option 1: live demo** — [TayyabManan/urdu-llm-chat](https://huggingface.co/spaces/TayyabManan/urdu-llm-chat). First call cold-starts Modal (~30-90s); subsequent calls 2-8s.

**Option 2: local inference** (need ~6 GB VRAM in 4-bit; 16 GB in bfloat16):

```python
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", max_seq_length=4096, load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "TayyabManan/qwen2.5-7b-urdu-v2")
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "پاکستان کا قومی ترانہ کس نے لکھا؟"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.7,
                         top_p=0.9, repetition_penalty=1.1, do_sample=True)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

See the [model card](https://huggingface.co/TayyabManan/qwen2.5-7b-urdu-v2) for recommended decoding per use case.

---

## Reproducibility

### Setup

```bash
git clone https://github.com/TayyabManan/Urdu-LLM
cd Urdu-LLM
python -m venv .venv && source .venv/bin/activate   # Linux/WSL
# or: python -m venv .venv && .venv\Scripts\activate # Windows
pip install -r requirements.txt
cp .env.example .env   # then edit
```

`.env` needs:
- `FT_ENDPOINT_URL` — Modal endpoint URL if you're using a hosted model
- `OPENAI_API_KEY` — only if you want to run the OpenAI judge
- `HF_TOKEN` — for pushing artifacts
- Modal auth set up via `modal token new` (separate, stored at `~/.modal.toml`)

### Train the model (Modal H100, ~$10-15)

```bash
modal run scripts/15_retrain_v2.py    # full v2 train, 2 epochs, ~5 hours on H100
# adapter lands at modal volume urdu-llm-vol:/7b-v2-adapter/
```

Hyperparameters in [`scripts/15_retrain_v2.py`](./scripts/15_retrain_v2.py). LoRA r=16, alpha=32,
all attention + MLP projection modules, effective batch 16, lr 2e-4 cosine.

### Generate eval responses (Modal H100, ~$1-2)

```bash
modal deploy scripts/19_serve_ft_endpoint.py   # spin up generation endpoint
python scripts/16_run_eval_v2.py               # writes data/eval/v2/outputs.jsonl
```

### Judge the eval (free with Claude Code CLI; ~$3-5 with OpenAI)

```bash
# Claude Code subprocess judge (no API key)
python scripts/18_run_judges.py --judges claude

# OpenAI judge (set OPENAI_API_KEY)
python scripts/18_run_judges.py --judges openai

# Both, for cross-model robustness
python scripts/18_run_judges.py --judges claude openai
```

Writes `data/eval/v2/judges/<judge>/judged_eval_results_v2.jsonl` per judge.

### Aggregate + plot

```bash
python scripts/17_aggregate_judges.py   # prints per-category, writes aggregate.png
```

### Tier 1.2 RAG (deferred to v3)

A RAG layer over Urdu Wikipedia was prototyped on top of v2 but didn't beat
plain v2-FT — the current FT model was trained on direct Q→A pairs only, so
prepended retrieved chunks push generation out of distribution. The fix is
upstream: v3 will include `(query, context, grounded-answer)` triples in the
training mix so the model actually learns to use retrieved context. RAG ships
properly with v3.

---

## Project structure

```
.
├── data/
│   ├── eval/                # 100-prompt eval set + base/FT/RAG outputs + judge verdicts
│   ├── code_roman/          # 1,671 LLM-generated code-mixed (Urdu/English) examples
│   ├── roman_urdu/          # Roman Urdu transliterations (v1)
│   └── roman_urdu_v2/       # Roman Urdu transliterations (v2)
├── prompts/                 # j2 templates: judge_v2.j2, rag_v1.j2
├── src/
│   ├── eval/                # Judging components: Claude Code subprocess, OpenAI, Haystack pipeline
│   └── rag/                 # Chunking + thin HTTP client to Modal /rag
├── scripts/
│   ├── 05_transliterate_to_roman.py  # data pipeline
│   ├── 12_generate_code_roman.py
│   ├── 15_retrain_v2.py              # v2 training entry
│   ├── 16_run_eval_v2.py             # eval generation
│   ├── 17_aggregate_judges.py        # FT-vs-base aggregator + plot
│   ├── 17b_aggregate_rag_vs_ft.py    # RAG-vs-FT aggregator
│   ├── 18_run_judges.py              # multi-judge eval driver (Haystack)
│   └── 19_serve_ft_endpoint.py       # Modal H100 server (/generate)
├── hf_publish/              # Files pushed to HuggingFace model repo (model card only)
├── spaces/urdu-llm-chat/    # Files pushed to HuggingFace Spaces (Gradio frontend)
└── urdu-llm-roadmap.md      # The original 30-day plan
```

---

## The journey

1. **v0 (Week 0):** Unsloth tutorial on Llama 3.2 1B + Qwen 2.5 0.5B to de-risk the pipeline before the public commitment.
2. **v1 (Week 1-2):** 3-epoch QLoRA on 50k Alpaca-translated examples. 51.5% wins, single judge. Code Mixed 0%, Code Explanation 10% — catastrophic forgetting on coding tasks because the training mix was Urdu-script-only.
3. **v2 (Week 3-4):** Added 1.6k code-mixed + 5k Roman Urdu examples. 2 epochs (not 3 — 3 overfit). Re-eval with 3 cross-model judges → **66% median, 91% creative, 80% translation + code-mixed**. Per-category regressions on summarisation/grammar/reasoning identified for v3.
4. **Tier 1.1 (Day 32):** Automated multi-judge evaluation pipeline. Claude Code subprocess via Max subscription replaces manual judging. Validated that auto judges score 8-14pp HIGHER than manual on the same outputs (position bias).
5. **Tier 1.2 RAG (deferred):** Prototyped a RAG layer; current FT model can't use prepended context (Q→A training only). Pivot: v3 will add RAG-format training data so the layer actually works.

Total cost: ~$50 of $80 budget ceiling.

---

## Tech stack

- **Base model:** [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Apache 2.0)
- **Fine-tune method:** QLoRA via [Unsloth](https://github.com/unslothai/unsloth)
- **Compute:** [Modal](https://modal.com) H100 (training + serving + indexing)
- **Vector store:** [Qdrant](https://qdrant.tech) (local-mode on Modal Volume) + [rank-bm25](https://github.com/dorianbrown/rank-bm25) sparse sidecar
- **Reranker:** [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) cross-encoder
- **Eval pipeline:** [Haystack 2.x](https://haystack.deepset.ai/) with custom Claude-Code-CLI and Modal-FT generators
- **Demo:** [Gradio](https://gradio.app) on HuggingFace Spaces (Modal endpoint as backend)
- **Tracking:** Weights & Biases for training loss / gradient norms

---

## Build-in-public

LinkedIn updates: [linkedin.com/in/tayyabmanan](https://linkedin.com/in/tayyabmanan)
Personal site: [tayyabmanan.com](https://tayyabmanan.com)

---

## Author

**Muhammad Tayyab** — MS Artificial Intelligence Engineering, COMSATS University
Islamabad. BS in Geographic Information Systems, University of the Punjab. 2 years
as AI Developer at Cointegration with hands-on multi-agent (LangChain / AutoGen /
CrewAI), RAG, and production LLM tooling experience.

This was my **first fine-tuning project end-to-end**, built in public over a
30-day window with weekly LinkedIn updates.

---

## License

- **Code:** Apache 2.0
- **Model weights:** Apache 2.0 (inherits from Qwen 2.5 base)
- **Eval prompts (`data/eval/prompts.jsonl`):** CC-BY-4.0 (use freely with attribution)
