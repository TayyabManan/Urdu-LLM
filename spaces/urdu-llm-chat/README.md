---
title: Urdu LLM Chat (Qwen 2.5 7B + v2 LoRA)
emoji: 💬
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Fine-tuned Qwen 2.5 7B for Urdu instruction-following
models:
- TayyabManan/qwen2.5-7b-urdu-v2
---

# Urdu LLM Chat

A Gradio demo for [Qwen 2.5 7B Urdu (v2 LoRA)](https://huggingface.co/TayyabManan/qwen2.5-7b-urdu-v2),
a QLoRA fine-tune of Qwen 2.5 7B Instruct on 63k Urdu instruction-response pairs.

**66% median pairwise preference vs base Qwen** across three independent LLM
judges on a 100-prompt evaluation set.

## How this Space works

The 7B model is too large for the free CPU tier (16 GB RAM, 2 vCPU, no GPU —
inference would be 30-60 s/token, effectively unusable). Instead, this Space is
a thin Gradio frontend that POSTs to a private Modal H100 endpoint where the
model actually runs. First call may take 30-90 s while the GPU container warms
up. Subsequent calls return in 2-8 s.

## Setup (Space owner)

1. In Space Settings → Variables and secrets, add a Secret:
   - Name: `FT_ENDPOINT_URL`
   - Value: `https://<your-modal-deployment>--ftserver-generate.modal.run`
2. Save. The Space will redeploy automatically.

## Acknowledgements

Built in public by [Muhammad Tayyab](https://linkedin.com/in/tayyabmanan). Full
training pipeline + evaluation code at [github.com/TayyabManan/Urdu-LLM](https://github.com/TayyabManan/Urdu-LLM).
