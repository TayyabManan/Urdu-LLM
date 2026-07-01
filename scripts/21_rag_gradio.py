"""Gradio side-by-side demo: plain v2-FT (no RAG) vs RAG-FT (hybrid+rerank).

Two columns of answers + retrieved sources panel. Designed for the Tier 1.2
writeup screencap — Balochistan-style hallucination-fix prompts show RAG's
single clear win; broader prompts show where RAG hurts.

    export FT_ENDPOINT_URL="https://digitization--urdu-rag-ft-endpoint-ftserver-generate.modal.run"
    python scripts/21_rag_gradio.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.rag.inference import build_rag_pipeline, query_rag, DEFAULT_SYSTEM  # noqa: E402


def call_generate(generate_url: str, prompt: str, max_tokens: int = 512,
                  temperature: float = 0.7, repetition_penalty: float = 1.1,
                  timeout_s: int = 180) -> str:
    """Plain /generate (no RAG context, uses FT adapter)."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "repetition_penalty": repetition_penalty,
    }
    with httpx.Client(timeout=timeout_s) as http:
        r = http.post(generate_url, json=payload)
        r.raise_for_status()
        return r.json().get("text", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--share", action="store_true",
                        help="Create a public gradio.live URL")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    import gradio as gr

    generate_url = os.environ.get("FT_ENDPOINT_URL")
    if not generate_url:
        raise SystemExit(
            "Set FT_ENDPOINT_URL in .env to the /generate URL, e.g.\n"
            "  https://digitization--urdu-rag-ft-endpoint-ftserver-generate.modal.run"
        )

    pipeline = build_rag_pipeline(
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        temperature=0.3,  # RAG uses low temp; plain FT uses --temperature
    )

    def respond(message: str):
        if not message or not message.strip():
            return "", "", "_(enter a query)_"

        ft_text = ""
        rag_text = ""
        sources_md = ""

        # Plain FT (no RAG)
        try:
            ft_text = call_generate(
                generate_url, message,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                repetition_penalty=1.1,
            )
        except Exception as e:
            ft_text = f"⚠️ /generate failed: {e!r}"

        # RAG-FT (hybrid+rerank)
        try:
            result = query_rag(pipeline, message, system=DEFAULT_SYSTEM)
            rag_text = result["answer"]
            sources = result["sources"]
            parts = []
            for i, s in enumerate(sources, start=1):
                title = s.get("title") or "(untitled)"
                url = s.get("url") or "#"
                score = s.get("score")
                snippet = (s.get("snippet") or "").strip()
                score_str = f" · rerank {score:.3f}" if isinstance(score, (int, float)) else ""
                parts.append(f"**[{i}] [{title}]({url}){score_str}**\n\n> {snippet}\n")
            sources_md = "\n---\n".join(parts) if parts else "_(no sources retrieved)_"
        except Exception as e:
            rag_text = f"⚠️ /rag failed: {e!r}"
            sources_md = "_(pipeline failed)_"

        return ft_text, rag_text, sources_md

    with gr.Blocks(
        title="Urdu RAG: plain FT vs RAG-FT",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Urdu LLM — RAG vs Plain Fine-Tune\n"
            "Qwen 2.5 7B + LoRA (v2 adapter) vs same model + Wikipedia-ur RAG "
            "(hybrid BM25+dense, bge-reranker-v2-m3, 310k chunks).\n\n"
            "_RAG fixes specific factual hallucinations (Balochistan example) "
            "but hurts overall (17% wins on 100-prompt eval). See [writeup]._"
        )
        with gr.Row():
            inp = gr.Textbox(
                label="Sawal (سوال) — ask in Urdu, Roman Urdu, or mixed",
                lines=2, scale=4,
            )
            btn = gr.Button("Pucho", variant="primary", scale=1)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Plain v2-FT (no retrieval)")
                out_ft = gr.Markdown(
                    value="_(awaiting query)_",
                    elem_id="plain-ft",
                )
            with gr.Column(scale=1):
                gr.Markdown("### RAG-FT (hybrid + rerank)")
                out_rag = gr.Markdown(
                    value="_(awaiting query)_",
                    elem_id="rag-ft",
                )
        with gr.Accordion("Retrieved Wikipedia sources (top 3, reranked)", open=True):
            out_sources = gr.Markdown(value="_(no query yet)_")

        btn.click(respond, inputs=inp, outputs=[out_ft, out_rag, out_sources])
        inp.submit(respond, inputs=inp, outputs=[out_ft, out_rag, out_sources])

        gr.Examples(
            examples=[
                ["پاکستان کا سب سے بڑا صوبہ رقبے کے لحاظ سے کون سا ہے؟"],
                ["مغل بادشاہ اکبر کی مذہبی پالیسی کیا تھی؟"],
                ["علامہ اقبال کا اصل نام کیا تھا؟"],
                ["Pakistan ka qaumi tarana kis ne likha?"],
                ["پاکستان کے 5 سب سے اونچے پہاڑ کون سے ہیں؟"],
            ],
            inputs=[inp],
            label="Demo prompts (factual — where RAG can help)",
        )

    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
