"""
Dry-run: Qwen 2.5 7B on 5,000 Urdu examples (Modal A100).

This proves the REAL model works with YOUR Urdu data before the full training run.
Same pipeline as 04_train.py but scaled up:
    - 7B model instead of 0.5B
    - 5,000 Urdu examples instead of 500 English
    - Urdu + Roman Urdu test prompts
    - 200 steps instead of 100

Run with:
    modal run scripts/06_train_7b_dryrun.py

Cost: ~$0.50-1.00 (A100 for ~15-30 min)
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
        gpu="A100",
    )
)

app = modal.App("7b-dryrun")
vol = modal.Volume.from_name("urdu-llm-vol", create_if_missing=True)

# Data lives on the Modal Volume at /vol/data/train.jsonl
# Upload it first with: modal run scripts/06a_upload_data.py


@app.function(
    gpu="A100",
    image=image,
    timeout=1800,           # 30 min — 7B is slower than 0.5B
    volumes={"/vol": vol},
)
def train():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import torch
    import json
    import random

    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # THE REAL MODEL
    MAX_SEQ_LENGTH = 2048
    N_EXAMPLES = 5000      # Dry run subset
    N_STEPS = 200          # Short run — just proving it works

    # ─── Load model ───
    print("=" * 50)
    print(f"Loading {MODEL_NAME} in 4-bit...")
    print("(This downloads ~4GB on first run, cached after)")
    print("=" * 50)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    free, total = torch.cuda.mem_get_info(0)
    print(f"Model loaded. VRAM: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.1f} GB")

    # ─── Attach LoRA ───
    print("\nAttaching LoRA adapters...")

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
    print(f"Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.2f}%)")

    # ─── Load formatted data ───
    # Data was uploaded to Modal Volume via 06a_upload_data.py
    # Already formatted with Qwen chat template — ready to use directly.
    print("\nLoading formatted Urdu data from Volume...")

    examples = []
    with open("/vol/data/train.jsonl", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Total available: {len(examples):,}")

    # Sample N_EXAMPLES for dry run
    random.seed(42)
    random.shuffle(examples)
    examples = examples[:N_EXAMPLES]
    print(f"Using {len(examples):,} for dry run")

    # Convert to HuggingFace Dataset format
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": [ex["text"] for ex in examples]})
    print(f"Dataset ready: {len(dataset)} examples")

    # ─── Test BEFORE training ───
    # Urdu, Roman Urdu, and code-mixed prompts — testing all three modes
    test_prompts = [
        # Urdu script
        ("پاکستان کی تاریخ کے بارے میں مختصر بتائیں۔", "Urdu script"),
        # Roman Urdu
        ("mujhe batao ke Python mein list comprehension kya hoti hai?", "Roman Urdu"),
        # Code-mixed
        ("democracy kya hai? simple words mein explain karo", "Code-mixed"),
        # Urdu creative
        ("بارش کے موسم پر ایک چھوٹی سی نظم لکھیں۔", "Urdu creative"),
    ]

    print("\n" + "=" * 50)
    print("BEFORE TRAINING — base model responses")
    print("=" * 50)

    FastLanguageModel.for_inference(model)
    for prompt, label in test_prompts:
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
        print(f"\n[{label}] >>> {prompt}")
        print(f"{response.strip()[:400]}")
        print("-" * 40)

    # ─── TRAIN ───
    print("\n" + "=" * 50)
    print(f"TRAINING — {N_STEPS} steps on {N_EXAMPLES:,} Urdu examples")
    print("=" * 50)
    print("Watch the loss — it should go DOWN.\n")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="/vol/7b-dryrun",

            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Effective batch = 8

            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,

            max_steps=N_STEPS,

            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),

            logging_steps=10,

            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=42,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
        ),
    )

    trainer_stats = trainer.train()

    print(f"\nTraining complete!")
    print(f"Steps: {trainer_stats.global_step}")
    print(f"Final loss: {trainer_stats.training_loss:.4f}")
    print(f"Time: {trainer_stats.metrics['train_runtime']:.0f} seconds")

    # ─── Test AFTER training ───
    print("\n" + "=" * 50)
    print("AFTER TRAINING — fine-tuned model responses")
    print("=" * 50)

    FastLanguageModel.for_inference(model)
    for prompt, label in test_prompts:
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
        print(f"\n[{label}] >>> {prompt}")
        print(f"{response.strip()[:400]}")
        print("-" * 40)

    # ─── Save adapter ───
    print("\n" + "=" * 50)
    print("Saving 7B adapter to Modal Volume...")
    print("=" * 50)

    ADAPTER_DIR = "/vol/7b-dryrun-adapter"
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    import os
    adapter_size = sum(
        os.path.getsize(os.path.join(ADAPTER_DIR, f))
        for f in os.listdir(ADAPTER_DIR)
        if f.endswith(('.safetensors', '.bin'))
    )
    print(f"Adapter saved to: {ADAPTER_DIR}")
    print(f"Adapter size: {adapter_size / 1024**2:.1f} MB")

    # ─── VRAM summary ───
    free, total = torch.cuda.mem_get_info(0)
    print(f"\nPeak VRAM: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.1f} GB")

    vol.commit()
    print("\nDone! 7B dry run complete on Modal.")


@app.local_entrypoint()
def main():
    print("Starting 7B dry-run training on Modal A100...")
    print("This will take ~15-30 minutes.")
    print("Cost: ~$0.50-1.00\n")
    train.remote()
    print("\n7B adapter saved to Modal Volume 'urdu-llm-vol' at /7b-dryrun-adapter")
