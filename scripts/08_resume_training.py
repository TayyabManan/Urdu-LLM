"""
Resume training from checkpoint-7500 (timed out at 7500/~10,092 steps).

Previous run (07_train_7b_full.py) timed out. Checkpoint saved at step 7500
on Modal Volume. This script resumes from that checkpoint to complete
the remaining ~2,600 steps.

Prerequisites:
    - Checkpoint exists at /vol/7b-full-checkpoints/checkpoint-7500
    - Data exists at /vol/data/train.jsonl

Run with:
    modal run scripts/08_resume_training.py
"""

import modal

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

app = modal.App("7b-resume-train")
vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048
ADAPTER_DIR = "/vol/7b-full-adapter"
LOG_FILE = "/vol/7b-full-adapter/resume_training_log.txt"
CHECKPOINT_DIR = "/vol/7b-full-checkpoints"


@app.function(
    gpu="H100",
    image=image,
    timeout=9200,          # 2 hours — plenty for ~2,600 remaining steps
    volumes={"/vol": vol},
)
def train():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import torch
    import json
    import os
    import glob

    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    # ─── Find latest checkpoint ───
    vol.reload()
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/checkpoint-*"))
    if not checkpoints:
        log("ERROR: No checkpoints found. Run 07_train_7b_full.py first.")
        return

    latest_ckpt = checkpoints[-1]
    log(f"Resuming from: {latest_ckpt}")

    # ─── Load model (same as original) ───
    log("=" * 60)
    log(f"Loading {MODEL_NAME} in 4-bit...")
    log("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    free, total = torch.cuda.mem_get_info(0)
    log(f"Model loaded. VRAM: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.1f} GB")

    # ─── Attach LoRA (same config) ───
    log("\nAttaching LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.2f}%)")

    # ─── Load data (same as original, same seed for identical split) ───
    log("\nLoading ALL formatted Urdu data from Volume...")

    examples = []
    with open("/vol/data/train.jsonl", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    log(f"Total examples loaded: {len(examples):,}")

    dataset = Dataset.from_dict({"text": [ex["text"] for ex in examples]})

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    log(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    n_steps_per_epoch = len(train_dataset) // 16
    n_total_steps = n_steps_per_epoch * 3
    log(f"Steps/epoch: ~{n_steps_per_epoch:,} | Total: ~{n_total_steps:,}")
    log(f"Remaining: ~{n_total_steps - 7500:,} steps")

    # ─── Test prompts ───
    test_prompts = [
        ("پاکستان کی تاریخ کے بارے میں مختصر بتائیں۔", "Urdu script"),
        ("mujhe batao ke Python mein list comprehension kya hoti hai?", "Roman Urdu"),
        ("democracy kya hai? simple words mein explain karo", "Code-mixed"),
        ("بارش کے موسم پر ایک چھوٹی سی نظم لکھیں۔", "Urdu creative"),
    ]

    def run_test_prompts(label):
        log(f"\n{'=' * 60}")
        log(f"{label}")
        log("=" * 60)

        FastLanguageModel.for_inference(model)
        responses = []
        for prompt, tag in test_prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, max_new_tokens=256, temperature=0.7,
                top_p=0.9, do_sample=True,
            )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            response = response.strip()[:400]
            log(f"\n[{tag}] >>> {prompt}")
            log(response)
            log("-" * 40)
            responses.append((tag, prompt, response))
        return responses

    # ─── TRAIN (resume from checkpoint) ───
    log(f"\n{'=' * 60}")
    log(f"RESUMING TRAINING from step 7500 — ~{n_total_steps - 7500:,} steps remaining")
    log(f"Effective batch: 16 | Eval every 500 steps | Checkpoints every 500 steps")
    log("=" * 60)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=CHECKPOINT_DIR,

            num_train_epochs=3,

            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,

            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,

            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),

            logging_steps=10,

            eval_strategy="steps",
            eval_steps=500,

            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,

            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",

            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=42,

            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
        ),
    )

    try:
        trainer_stats = trainer.train(resume_from_checkpoint=True)

        log(f"\nTraining complete!")
        log(f"Steps: {trainer_stats.global_step}")
        log(f"Final train loss: {trainer_stats.training_loss:.4f}")
        log(f"Time: {trainer_stats.metrics['train_runtime']:.0f} seconds")
        log(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.1f}")

        after_responses = run_test_prompts("AFTER TRAINING — fine-tuned model responses")

        # ─── Save adapter ───
        log(f"\n{'=' * 60}")
        log("Saving adapter to Modal Volume...")
        log("=" * 60)

        os.makedirs(ADAPTER_DIR, exist_ok=True)
        model.save_pretrained(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)

        adapter_size = sum(
            os.path.getsize(os.path.join(ADAPTER_DIR, f))
            for f in os.listdir(ADAPTER_DIR)
            if f.endswith((".safetensors", ".bin"))
        )
        log(f"Adapter saved to: {ADAPTER_DIR}")
        log(f"Adapter size: {adapter_size / 1024**2:.1f} MB")

        # ─── VRAM summary ───
        peak = torch.cuda.max_memory_allocated(0)
        total_mem = torch.cuda.get_device_properties(0).total_mem
        log(f"\nPeak VRAM allocated: {peak / 1024**3:.2f} GB / {total_mem / 1024**3:.1f} GB")

    except Exception as e:
        log(f"\nTRAINING FAILED: {e}")
        import traceback
        log(traceback.format_exc())

    finally:
        os.makedirs(ADAPTER_DIR, exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"Training log saved to: {LOG_FILE}")

        vol.commit()
        print("\nVolume committed. Done.")


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("RESUME: Qwen 2.5 7B — completing from checkpoint-7500")
    print("=" * 60)
    print(f"GPU: H100 | ~2,600 steps remaining")
    print(f"Estimated: ~30-45 min | Cost: ~$1-2")
    print()
    train.remote()
    print("\nAdapter saved to Modal Volume 'urdu-llm-vol' at /7b-full-adapter")
    print("Training log at /7b-full-adapter/resume_training_log.txt")
