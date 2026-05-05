"""
Run formal evaluation: generate responses from base and fine-tuned models
on 100 hand-curated Urdu prompts.

Loads base Qwen 2.5 7B, runs all prompts, unloads, then loads fine-tuned
(base + LoRA adapter) and runs same prompts. Identical decoding params.

Saves results to Modal Volume as eval_results_v2.jsonl.

Prerequisites:
    - Adapter at /vol/7b-v2-adapter/ on Modal Volume
    - Eval prompts at data/eval/eval_prompts.jsonl (uploaded by this script)

Run with:
    modal run scripts/16_run_eval_v2.py
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

app = modal.App("urdu-eval")
vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048
ADAPTER_DIR = "/vol/7b-v2-adapter"
RESULTS_FILE = "/vol/eval/eval_results_v2.jsonl"

# Decoding params from training_config.yaml
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 512


@app.function(
    gpu="H100",
    image=image,
    timeout=7200,
    volumes={"/vol": vol},
)
def run_eval(prompts: list[dict]):
    from unsloth import FastLanguageModel
    from peft import PeftModel
    import torch
    import os
    import gc
    import time

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
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        return response.strip()

    vol.reload()

    # ═══════════════════════════════════════════════
    # Phase 1: Base model responses
    # ═══════════════════════════════════════════════
    print("=" * 60)
    print(f"Phase 1: Loading BASE model ({MODEL_NAME})...")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.1f} GB")

    base_responses = {}
    start = time.time()
    for i, p in enumerate(prompts):
        resp = generate_response(model, tokenizer, p["prompt"])
        base_responses[p["id"]] = resp
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  Base: {i+1}/{len(prompts)} done ({elapsed:.0f}s)")

    print(f"Base model: {len(base_responses)} responses in {time.time() - start:.0f}s")

    # Free base model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════
    # Phase 2: Fine-tuned model responses
    # ═══════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"Phase 2: Loading FINE-TUNED model (base + adapter)...")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    FastLanguageModel.for_inference(model)

    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.1f} GB")

    ft_responses = {}
    start = time.time()
    for i, p in enumerate(prompts):
        resp = generate_response(model, tokenizer, p["prompt"])
        ft_responses[p["id"]] = resp
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  Fine-tuned: {i+1}/{len(prompts)} done ({elapsed:.0f}s)")

    print(f"Fine-tuned model: {len(ft_responses)} responses in {time.time() - start:.0f}s")

    # ═══════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════
    os.makedirs("/vol/eval", exist_ok=True)

    results = []
    for p in prompts:
        results.append({
            "id": p["id"],
            "category": p["category"],
            "language": p["language"],
            "prompt": p["prompt"],
            "base_response": base_responses[p["id"]],
            "finetuned_response": ft_responses[p["id"]],
        })

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults saved: {RESULTS_FILE} ({len(results)} entries)")

    # Print sample comparisons
    print(f"\n{'=' * 60}")
    print("SAMPLE COMPARISONS (first 5)")
    print("=" * 60)
    for r in results[:5]:
        print(f"\n[{r['category']}] {r['prompt'][:80]}...")
        print(f"  BASE: {r['base_response'][:200]}...")
        print(f"  FINE: {r['finetuned_response'][:200]}...")
        print("-" * 40)

    vol.commit()
    print("\nDone. Volume committed.")
    return results


@app.local_entrypoint()
def main():
    # Load eval prompts from local file
    prompts_path = Path(__file__).parent.parent / "data" / "eval" / "eval_prompts.jsonl"
    prompts = []
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))

    print("=" * 60)
    print(f"Formal Eval: {len(prompts)} prompts")
    print("=" * 60)

    categories = {}
    for p in prompts:
        categories[p["category"]] = categories.get(p["category"], 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print(f"\nDecoding: temp={TEMPERATURE}, top_p={TOP_P}, max_tokens={MAX_NEW_TOKENS}")
    print(f"Models: base ({MODEL_NAME}) vs fine-tuned (adapter)")
    print()

    results = run_eval.remote(prompts)

    # Save locally for judging
    local_path = Path(__file__).parent.parent / "data" / "eval" / "eval_results.jsonl"
    with open(local_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nEval complete: {len(results)} prompt-response pairs")
    print(f"Saved locally: {local_path}")
    print(f"Also on Modal Volume: /vol/eval/eval_results.jsonl")
