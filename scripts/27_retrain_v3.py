~"""
Retrain v3: Qwen 2.5 7B on ~69k examples (the v2 set + RAG triples + grammar + summ/reason).

Changes from v2 (scripts/15_retrain_v2.py):
    - MAX_SEQ_LENGTH 2048 -> 4096. v3 adds RAG examples that carry retrieved context; the
      combine (scripts/26) caps RAG at <=3900 tokens, so nothing truncates at 4096.
    - Reads /vol/data/train_v3.jsonl (v2's /vol/data/train.jsonl is left untouched).
    - Saves the adapter to /vol/7b-v3-adapter/ (separate from v2).
    - +5,675 v3 synthetic rows: RAG-aware (trained on the EXACT {chunks}\\n\\nسوال:{query}
      surface the /rag endpoint serves), grammar-correction, summarization, and reasoning —
      to fix the v2 regressions (summ/reasoning/grammar) and make RAG actually work.
    - Same recipe otherwise: 2 epochs, effective batch 16, save_total_limit=5,
      load_best_model_at_end on eval_loss.

Run with:
    modal run scripts/27_retrain_v3.py
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

app = modal.App("7b-v3-train")
vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 4096          # v2 was 2048; v3 RAG rows carry context (<=3900 tokens)
DATA_FILE = "/vol/data/train_v3.jsonl"
ADAPTER_DIR = "/vol/7b-v3-adapter"
CHECKPOINT_DIR = "/vol/7b-v3-checkpoints"
LOG_FILE = "/vol/7b-v3-adapter/training_log.txt"


@app.function(
    gpu="H100",
    image=image,
    timeout=30000,
    volumes={"/vol": vol},
)
def train():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import torch
    import json
    import os

    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    vol.reload()

    # ─── Load model ───
    log("=" * 60)
    log(f"Loading {MODEL_NAME} in 4-bit (max_seq_length={MAX_SEQ_LENGTH})...")
    log("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    free, total = torch.cuda.mem_get_info(0)
    log(f"Model loaded. VRAM: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.1f} GB")

    # ─── Attach LoRA ───
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

    # ─── Load data ───
    log(f"\nLoading v3 training data from {DATA_FILE}...")

    examples = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    log(f"Total examples loaded: {len(examples):,}")

    dataset = Dataset.from_dict({"text": [ex["text"] for ex in examples]})

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    log(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    n_steps_per_epoch = len(train_dataset) // 16
    n_total_steps = n_steps_per_epoch * 2
    log(f"Steps/epoch: ~{n_steps_per_epoch:,} | Total (2 epochs): ~{n_total_steps:,}")

    # ─── Test prompts (general sanity before/after; the real eval is scripts/28) ───
    test_prompts = [
        ("پاکستان کی تاریخ کے بارے میں مختصر بتائیں۔", "Urdu script"),
        ("mujhe batao ke Python mein list comprehension kya hoti hai?", "Roman Urdu"),
        ("democracy kya hai? simple words mein explain karo", "Code-mixed"),
        ("بارش کے موسم پر ایک چھوٹی سی نظم لکھیں۔", "Urdu creative"),
        ("ایک دکاندار نے قمیض 700 روپے میں خرید کر 950 میں بیچی۔ منافع کتنا ہوا؟", "Reasoning"),
        ("اس متن کا ایک جملے میں خلاصہ کریں:\n\nپاکستان جنوبی ایشیا کا ایک ملک ہے جس کا "
         "دارالحکومت اسلام آباد ہے اور یہ 1947 میں قائم ہوا۔", "Summarization"),
    ]

    def run_test_prompts(label):
        log(f"\n{'=' * 60}")
        log(f"{label}")
        log("=" * 60)

        FastLanguageModel.for_inference(model)
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

    run_test_prompts("BEFORE TRAINING — base model responses")

    # ─── TRAIN ───
    log(f"\n{'=' * 60}")
    log(f"TRAINING v3 — 2 epochs on {len(train_dataset):,} examples")
    log(f"Effective batch: 16 | Eval every 500 steps | Checkpoints every 500 steps")
    log(f"max_seq_length={MAX_SEQ_LENGTH} | save_total_limit=5 (best-by-eval-loss)")
    log("=" * 60)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=CHECKPOINT_DIR,

            num_train_epochs=2,

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
            save_total_limit=5,

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

    # ─── Resume from checkpoint if available ───
    resume_ckpt = None
    if os.path.isdir(CHECKPOINT_DIR):
        ckpts = sorted(
            [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if ckpts:
            resume_ckpt = os.path.join(CHECKPOINT_DIR, ckpts[-1])
            log(f"Resuming from checkpoint: {resume_ckpt}")

    try:
        trainer_stats = trainer.train(resume_from_checkpoint=resume_ckpt)

        log(f"\nTraining complete!")
        log(f"Steps: {trainer_stats.global_step}")
        log(f"Final train loss: {trainer_stats.training_loss:.4f}")
        log(f"Time: {trainer_stats.metrics['train_runtime']:.0f} seconds")
        log(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.1f}")

        run_test_prompts("AFTER TRAINING — fine-tuned v3 responses")

        # ─── Save adapter ───
        log(f"\n{'=' * 60}")
        log("Saving v3 adapter to Modal Volume...")
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

        peak = torch.cuda.max_memory_allocated(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        log(f"\nPeak VRAM: {peak / 1024**3:.2f} GB / {total_mem / 1024**3:.1f} GB")

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
    print("v3 QLoRA: Qwen 2.5 7B x ~69k examples (2 epochs)")
    print("=" * 60)
    print(f"GPU: H100 | Epochs: 2 | Effective batch: 16 | max_seq_length: {MAX_SEQ_LENGTH}")
    print(f"New vs v2: +RAG triples (RAG-aware), +grammar, +summ/reasoning fixes")
    print(f"Data: {DATA_FILE} | Adapter -> {ADAPTER_DIR}")
    print(f"Estimated: ~2-3 hours | Cost: ~$5-8")
    print(f"Checkpoints every 500 steps (keep 5)")
    print()
    train.remote()
    print(f"\nv3 adapter saved to Modal Volume at {ADAPTER_DIR}")
    print(f"Training log at {LOG_FILE}")
