"""
Check eval loss curve from trainer_state.json on Modal Volume.

Prints table of train/eval loss at each eval step.
Looks for overfitting (eval loss rising while train loss drops).

Run with:
    modal run scripts/09_check_eval_loss.py
"""

import modal

app = modal.App("check-eval-loss")
vol = modal.Volume.from_name("urdu-llm-vol")


@app.function(
    volumes={"/vol": vol},
)
def read_trainer_state():
    import json
    import glob

    vol.reload()

    # Read ALL trainer_state.json files and merge log histories.
    # Original run and resume run may have separate checkpoints.
    candidates = sorted(glob.glob("/vol/7b-full-checkpoints/checkpoint-*/trainer_state.json"))
    if not candidates:
        return None

    merged = {}
    paths_loaded = []
    for path in candidates:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        paths_loaded.append(path)
        for entry in state["log_history"]:
            key = (entry["step"], "eval" if "eval_loss" in entry else "train")
            if key not in merged:
                merged[key] = entry

    log_history = sorted(merged.values(), key=lambda e: e["step"])

    return {"paths": paths_loaded, "log_history": log_history}


def plot_losses(log_history, out_dir="docs/training_plots"):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(out_dir, exist_ok=True)

    eval_entries = [e for e in log_history if "eval_loss" in e]
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]

    train_steps = [e["step"] for e in train_entries]
    train_losses = [e["loss"] for e in train_entries]
    eval_steps = [e["step"] for e in eval_entries]
    eval_losses = [e["eval_loss"] for e in eval_entries]

    # Epoch boundaries
    steps_per_epoch = max(train_steps) / 3 if train_steps else 0

    # --- Plot 1: Train + Eval loss ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_steps, train_losses, alpha=0.4, linewidth=0.8, color="#2196F3", label="Train loss")
    ax.plot(eval_steps, eval_losses, linewidth=2, color="#F44336", marker="o", markersize=4, label="Eval loss")

    for i in range(1, 4):
        ax.axvline(x=steps_per_epoch * i, color="gray", linestyle="--", alpha=0.4)
        ax.text(steps_per_epoch * i, ax.get_ylim()[1] * 0.95, f"Epoch {i}", ha="center", fontsize=9, color="gray")

    best = min(eval_entries, key=lambda e: e["eval_loss"])
    ax.annotate(f"Best: {best['eval_loss']:.4f}", xy=(best["step"], best["eval_loss"]),
                xytext=(best["step"] + 300, best["eval_loss"] + 0.1),
                arrowprops=dict(arrowstyle="->", color="#4CAF50"), fontsize=10, color="#4CAF50", fontweight="bold")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Urdu QLoRA Fine-Tuning — Train vs Eval Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/train_eval_loss.png", dpi=150)
    print(f"Saved: {out_dir}/train_eval_loss.png")
    plt.close(fig)

    # --- Plot 2: Eval loss zoomed ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_steps, eval_losses, linewidth=2, color="#F44336", marker="o", markersize=5)

    for i, (s, l) in enumerate(zip(eval_steps, eval_losses)):
        ax.annotate(f"{l:.4f}", xy=(s, l), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=7, color="#666")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Eval Loss", fontsize=12)
    ax.set_title("Eval Loss per 500 Steps", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/eval_loss_zoomed.png", dpi=150)
    print(f"Saved: {out_dir}/eval_loss_zoomed.png")
    plt.close(fig)

    # --- Plot 3: Learning rate schedule ---
    lr_entries = [e for e in log_history if "learning_rate" in e]
    if lr_entries:
        fig, ax = plt.subplots(figsize=(10, 4))
        lr_steps = [e["step"] for e in lr_entries]
        lr_vals = [e["learning_rate"] for e in lr_entries]
        ax.plot(lr_steps, lr_vals, linewidth=1.5, color="#9C27B0")
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Cosine LR Schedule", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/learning_rate.png", dpi=150)
        print(f"Saved: {out_dir}/learning_rate.png")
        plt.close(fig)

    # --- Plot 4: Train loss smoothed ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_steps, train_losses, alpha=0.2, linewidth=0.5, color="#2196F3", label="Raw")

    # EMA smoothing
    window = 50
    if len(train_losses) > window:
        smoothed = []
        val = train_losses[0]
        alpha = 2 / (window + 1)
        for l in train_losses:
            val = alpha * l + (1 - alpha) * val
            smoothed.append(val)
        ax.plot(train_steps, smoothed, linewidth=2, color="#1565C0", label=f"EMA (window={window})")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_title("Train Loss — Smoothed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/train_loss_smoothed.png", dpi=150)
    print(f"Saved: {out_dir}/train_loss_smoothed.png")
    plt.close(fig)


@app.local_entrypoint()
def main():
    result = read_trainer_state.remote()

    if result is None:
        print("No trainer_state.json found in any checkpoint.")
        return

    for p in result["paths"]:
        print(f"Loaded: {p}")
    print()

    log_history = result["log_history"]

    eval_entries = [e for e in log_history if "eval_loss" in e]
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]

    # ─── Table ───
    print(f"{'Step':>6} | {'Train Loss':>11} | {'Eval Loss':>10} | {'Note'}")
    print("-" * 55)

    prev_eval = None
    for e in eval_entries:
        step = e["step"]
        eval_loss = e["eval_loss"]

        train = [t for t in train_entries if t["step"] <= step]
        train_loss = f"{train[-1]['loss']:.4f}" if train else "—"

        note = ""
        if prev_eval is not None:
            if eval_loss > prev_eval + 0.01:
                note = "<< EVAL RISING"
            elif eval_loss < prev_eval:
                note = "improving"
            else:
                note = "plateau"

        print(f"{step:>6} | {train_loss:>11} | {eval_loss:.4f}     | {note}")
        prev_eval = eval_loss

    print("-" * 55)
    if eval_entries:
        best = min(eval_entries, key=lambda e: e["eval_loss"])
        print(f"Best eval loss: {best['eval_loss']:.4f} at step {best['step']}")

        last = eval_entries[-1]
        if last["eval_loss"] > best["eval_loss"] + 0.02:
            print(f"WARNING: Final eval loss ({last['eval_loss']:.4f}) higher than best ({best['eval_loss']:.4f})")
            print(f"Model may be overfitting. load_best_model_at_end=True should have picked best checkpoint.")
        else:
            print("No significant overfitting detected.")

    # ─── Plots ───
    print("\nGenerating plots...")
    plot_losses(log_history)
