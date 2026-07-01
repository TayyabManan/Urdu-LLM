"""
v3 eval generation: generate the v3 fine-tuned model's responses on the 100 eval prompts,
REUSING the v2 base_response (the base model is unchanged across versions).

Fork of scripts/16_run_eval_v2.py. Differences:
  - adapter /vol/7b-v3-adapter (not v2)
  - SKIPS base-model generation — reuses base_response from data/eval/v2/outputs.jsonl
  - reads the prompts FROM data/eval/v2/outputs.jsonl too (guarantees identical ids /
    categories, so the base-reuse join and the later v3-vs-v2 comparison line up)
  - writes data/eval/v3/outputs.jsonl (same schema) + /vol/eval/eval_results_v3.jsonl

Run with:
    modal run scripts/28_run_eval_v3.py
"""

import modal
import json
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "unsloth",
        "xformers",
    )
    .pip_install(
        "trl>=0.15",
        "peft",
        "accelerate",
        "bitsandbytes",
        "transformers>=4.46",
        gpu="H100",
    )
)

app = modal.App("urdu-eval-v3")
vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 4096          # match v3 training (eval prompts are short anyway)
ADAPTER_DIR = "/vol/7b-v3-adapter"
RESULTS_FILE = "/vol/eval/eval_results_v3.jsonl"

# Decoding params — identical to v2 (scripts/16) so the comparison is apples-to-apples.
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 512


@app.function(
    gpu="H100",
    image=image,
    timeout=7200,
    volumes={"/vol": vol},
)
def run_eval_v3(prompts: list[dict], base_responses: dict):
    from unsloth import FastLanguageModel
    from peft import PeftModel
    import torch
    import os
    import time

    # base_responses arrives JSON-style (str keys) — normalize to int ids.
    base_by_id = {int(k): v for k, v in base_responses.items()}

    def generate_response(model, tokenizer, prompt_text):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
        )
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()

    vol.reload()

    print("=" * 60)
    print(f"Loading FINE-TUNED v3 model (base + {ADAPTER_DIR})...")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    FastLanguageModel.for_inference(model)

    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.1f} GB")

    ft_responses = {}
    start = time.time()
    for i, p in enumerate(prompts):
        ft_responses[p["id"]] = generate_response(model, tokenizer, p["prompt"])
        if (i + 1) % 10 == 0:
            print(f"  v3 FT: {i+1}/{len(prompts)} done ({time.time() - start:.0f}s)")
    print(f"v3 FT: {len(ft_responses)} responses in {time.time() - start:.0f}s")

    os.makedirs("/vol/eval", exist_ok=True)
    results = []
    missing_base = 0
    for p in prompts:
        base = base_by_id.get(p["id"], "")
        if not base:
            missing_base += 1
        results.append({
            "id": p["id"],
            "category": p["category"],
            "language": p.get("language", ""),
            "prompt": p["prompt"],
            "base_response": base,                       # reused from v2 (base unchanged)
            "finetuned_response": ft_responses[p["id"]],
        })
    if missing_base:
        print(f"WARNING: {missing_base} prompts had no reused base_response")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {RESULTS_FILE} ({len(results)} entries)")

    print(f"\n{'=' * 60}\nSAMPLE (first 3)\n{'=' * 60}")
    for r in results[:3]:
        print(f"\n[{r['category']}] {r['prompt'][:70]}...")
        print(f"  v3 FT: {r['finetuned_response'][:180]}...")

    vol.commit()
    print("\nDone. Volume committed.")
    return results


@app.local_entrypoint()
def main():
    root = Path(__file__).parent.parent
    v2_out = root / "data" / "eval" / "v2" / "outputs.jsonl"
    if not v2_out.exists():
        raise SystemExit(f"Missing {v2_out} (need it for the prompts + reused base_response).")

    rows = [json.loads(l) for l in open(v2_out, encoding="utf-8") if l.strip()]
    prompts = [
        {"id": r["id"], "category": r["category"],
         "language": r.get("language", ""), "prompt": r["prompt"]}
        for r in rows
    ]
    base_responses = {str(r["id"]): r.get("base_response", "") for r in rows}

    print("=" * 60)
    print(f"v3 eval: {len(prompts)} prompts | reusing {len(base_responses)} v2 base_responses")
    print(f"Adapter: {ADAPTER_DIR} | decoding temp={TEMPERATURE} top_p={TOP_P} max_tokens={MAX_NEW_TOKENS}")
    print("=" * 60)

    results = run_eval_v3.remote(prompts, base_responses)

    out_dir = root / "data" / "eval" / "v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "outputs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved locally: {out_path} ({len(results)} rows)")
    print("Next: judges (scripts/18) for v3-vs-base + v3-vs-v2.")
