---
language:
- ur
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
tags:
- urdu
- qwen
- qwen2.5
- lora
- qlora
- peft
- instruction-tuning
- low-resource-language
- pakistan
pipeline_tag: text-generation
model-index:
- name: qwen2.5-7b-urdu-v2
  results:
  - task:
      type: text-generation
      name: Urdu Instruction Following (pairwise vs base)
    metrics:
    - type: win_rate_vs_base
      value: 66.0
      name: Median pairwise preference vs base (3 LLM judges)
---

# Qwen 2.5 7B Urdu (v2 LoRA adapter)

QLoRA fine-tune of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
specialised on Urdu instruction-following. Trained on 63,322 instruction-response pairs
covering Urdu script and Roman Urdu, with explicit code-mixed (Urdu-English) and
Pakistani-context examples.

**This repo ships the LoRA adapter only (~154 MB).** Apply on top of the base model with PEFT
to use. See the **Usage** section below.

Built as a **30-day public build** by a first-time fine-tuner. Full training
pipeline + eval scripts + roadmap on [GitHub](https://github.com/TayyabManan/Urdu-LLM).

---

## Headline result

**Median 66% pairwise preference vs base Qwen 2.5 7B Instruct across three independent
LLM judges**, on a 100-prompt hand-curated Urdu evaluation set. Range 48–67% across judges.

| Judge | Win rate vs base |
|---|---|
| Claude Desktop (Opus 4.7) | 67.0% |
| GPT 5.3 Thinking | 66.0% |
| Gemini 3.1 Pro | 48.0% |
| **Median (3 judges)** | **66.0%** |
| Claude Code (Opus 4.7) — auto, position-randomized | 65.0% |

A fourth auto-judge run via the Claude Code CLI subprocess (no API key) was excluded from
the median because of 79% same-model agreement with Claude Desktop. The cross-model
judges (GPT + Gemini) are the meaningful stress test.

### Per-category breakdown (median across 3 judges)

| Category | v2 win rate vs base | Direction |
|---|---|---|
| Creative writing | 91% | strong gain |
| Code-mixed (Urdu/English) | 80% | strong gain (v1 was 0%) |
| Translation (UR↔EN) | 80% | strong gain |
| Question Answering | 71% | gain |
| Code explanation in Urdu | 60% | gain |
| Summarization | 46% | **regression** (v1 was 73%) |
| Reasoning | 31% | **regression** (v1 was 44%) |
| Grammar correction | 36% | **regression** (v1 was 40%) |

The regressions on summarization / reasoning / grammar are the v3 work-list (see
[GitHub roadmap](https://github.com/TayyabManan/Urdu-LLM/blob/main/urdu-llm-roadmap.md)).

---

## Quick start (Usage)

**Recommended: load via Unsloth** (the adapter was trained against
`unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit` — see `adapter_config.json`).
Unsloth gives identical-base loading and ~2× inference speed on consumer GPUs.

```python
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "TayyabManan/qwen2.5-7b-urdu-v2")
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "پاکستان کا قومی ترانہ کس نے لکھا؟"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
import torch
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9,
                         repetition_penalty=1.1, do_sample=True)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### Alternative: vanilla Transformers + PEFT

Works, but the adapter was calibrated against Unsloth's 4-bit base — expect
minor numerical drift (no catastrophic failure).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(model, "TayyabManan/qwen2.5-7b-urdu-v2")
model.eval()
```

### Recommended decoding

| Use case | temp | top_p | rep_penalty | max_new_tokens |
|---|---|---|---|---|
| Creative / open-ended | 0.7 | 0.9 | 1.1 | 512 |
| Factual / short answer | 0.3 | 0.9 | 1.1 | 256 |
| Long Urdu generation (avoid repetition collapse) | 0.5 | 0.9 | 1.3 | 512 |

---

## Training details

- **Base model:** Qwen/Qwen2.5-7B-Instruct (Apache 2.0)
- **Method:** QLoRA (4-bit NF4 base weights + LoRA)
- **LoRA:** r=16, alpha=32, targets = q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
- **Effective batch:** 16 (per-device 2 × grad_accum 8)
- **Optimiser:** AdamW 8-bit, lr=2e-4, cosine schedule, 3% warmup
- **Epochs:** 2 (v1 was 3 — over-trained, overfit; 2 is the sweet spot for this data mix)
- **Max sequence length:** 2048 tokens (training); 4096 (inference, after extension)
- **Precision:** bfloat16 forward, 4-bit NF4 base
- **Hardware:** Single NVIDIA H100 80GB on Modal
- **Wall-clock:** 7,520 steps, ~5 hours
- **Final train loss:** 0.3699
- **Peak VRAM:** 8.56 GB
- **Framework:** [Unsloth](https://github.com/unslothai/unsloth) (2× throughput vs stock TRL on H100)
- **Adapter size:** 154 MB
- **Cost (single run):** ~$10-15 on Modal H100 (~$3.50/hr)

### Training data (63,322 examples total)

| Source | Count | Type |
|---|---|---|
| Translated instruction data (Urdu script) | 56,666 | Q→A, multi-domain |
| Code Roman synthetic (LLM-generated) | 1,671 | Code-mixed instruction following |
| Roman Urdu v2 (LLM-transliterated from Urdu) | 4,985 | Roman-script Urdu Q→A |

All data formatted as Alpaca-style JSONL: `{"instruction": ..., "input": ..., "output": ...}`.
Qwen chat template applied at training time, not at data-preparation time, for flexibility.

**Important:** This training mix is **direct Q→A pairs only**. There are NO `(query, context, grounded_answer)`
triples — which is why Tier 1.2 RAG over Urdu Wikipedia did not improve on this model (the FT
distribution doesn't include prepended context). Future v3 will add RAG-format examples.

---

## Limitations

- **English-leaning system prompt sensitivity.** Training used `"You are a helpful assistant."`
  exactly. Long or Urdu system prompts can destabilise generation.
- **Length-constraint instructions are weak.** Summarisation regressed from v1's 73% to 46% —
  the model frequently ignores "in 3 sentences" / "in one paragraph" prompts.
- **Repetition collapse on long Urdu prompts** (>2k tokens). Mitigations: keep
  `max_new_tokens ≤ 512`, raise `repetition_penalty` to 1.3, lower `temperature` to 0.3-0.5.
- **No RAG support out of the box.** This v2 adapter was trained on direct Q→A pairs only,
  so prepending retrieved context pushes generation out of distribution. RAG-aware training
  is on the v3 roadmap.
- **Not for safety-critical use.** Standard refusal behaviours from the base model are
  preserved but not strengthened.
- **Eval is pairwise preference, not absolute capability.** A 66% win vs base does not imply
  68% of responses are factually correct. Many wins are stylistic / fluency-based.

---

## Evaluation methodology

- **Set:** 100 hand-curated Urdu prompts across 8 categories (QA, summarisation, translation,
  grammar, reasoning, creative, code explanation, code-mixed). Authored by the project author;
  spot-validated by one other native Urdu speaker.
- **Decoding parity:** Both base and fine-tuned generate with identical decoding parameters
  (temp 0.7, top-p 0.9, max 512 tokens, rep_penalty 1.1).
- **Judging:** Pairwise blinded preference. Responses A/B-randomised per item to control
  position bias.
- **Judges:** 3 independent LLM judges (Claude Desktop / GPT 5.3 / Gemini 3.1 Pro) plus an
  automated subprocess judge via Claude Code CLI.
- **Aggregation:** Per-category and overall win-rates per judge; **median across 3 judges**
  reported as the headline number.

Eval code: [scripts/16_run_eval_v2.py](https://github.com/TayyabManan/Urdu-LLM/blob/main/scripts/16_run_eval_v2.py)
+ [scripts/18_run_judges.py](https://github.com/TayyabManan/Urdu-LLM/blob/main/scripts/18_run_judges.py)
+ [scripts/17_aggregate_judges.py](https://github.com/TayyabManan/Urdu-LLM/blob/main/scripts/17_aggregate_judges.py).

---

## Acknowledgements

- Base model: **Qwen team** at Alibaba ([Qwen 2.5 paper](https://arxiv.org/abs/2412.15115))
- Training framework: **Unsloth** by Daniel Han + team
- Serverless compute: **Modal**
- Evaluation infrastructure: **Haystack** by deepset
- Data inspiration: **Aya Dataset** from Cohere for AI

---

## Author

**Muhammad Tayyab** — MS Artificial Intelligence Engineering, COMSATS University Islamabad.
2yr as AI Developer at Cointegration.

- GitHub: [TayyabManan](https://github.com/TayyabManan)
- LinkedIn: [linkedin.com/in/tayyabmanan](https://linkedin.com/in/tayyabmanan)
- Web: [tayyabmanan.com](https://tayyabmanan.com)

This is my first fine-tuning project end-to-end. Total project cost to date: ~$50.

---

## Citation

```bibtex
@misc{tayyab2026qwen25urduv2,
  title  = {Qwen 2.5 7B Urdu (v2 LoRA adapter)},
  author = {Tayyab, Muhammad},
  year   = {2026},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/TayyabManan/qwen2.5-7b-urdu-v2}
}
```

## License

Apache 2.0. Inherits from Qwen 2.5 base.
