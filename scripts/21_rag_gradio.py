"""Gradio chat UI for the Urdu RAG pipeline.

Sets up `build_rag_pipeline()` once, serves a chat interface that shows the
answer + retrieved Wikipedia sources side-by-side.

Usage:
    export FT_ENDPOINT_URL="https://digitization--urdu-rag-ft-endpoint-ftserver-generate.modal.run"
    python scripts/21_rag_gradio.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.rag.inference import build_rag_pipeline, query_rag, DEFAULT_SYSTEM  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embed-device", default=None,
                        help="cpu | cuda | mps. None=auto")
    parser.add_argument("--share", action="store_true",
                        help="Create a public gradio.live URL")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    import gradio as gr

    if not os.environ.get("FT_ENDPOINT_URL"):
        raise SystemExit(
            "Set FT_ENDPOINT_URL in .env (e.g. "
            "https://digitization--urdu-rag-ft-endpoint-ftserver-generate.modal.run)"
        )

    print("Building RAG pipeline...")
    pipeline = build_rag_pipeline(
        top_k=args.top_k,
        embed_device=args.embed_device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print("Pipeline ready.")

    def respond(message: str, history: list):
        try:
            result = query_rag(pipeline, message, system=DEFAULT_SYSTEM)
        except Exception as e:
            return f"⚠️ Error: {e!r}", "(no sources — pipeline failed)"
        answer = result["answer"]
        sources_md_parts = []
        for i, s in enumerate(result["sources"], start=1):
            title = s.get("title") or "(untitled)"
            url = s.get("url") or "#"
            score = s.get("score")
            snippet = s.get("snippet", "").strip()
            score_str = f" · score {score:.3f}" if isinstance(score, (int, float)) else ""
            sources_md_parts.append(
                f"**[{i}] [{title}]({url}){score_str}**\n\n> {snippet}\n"
            )
        sources_md = "\n---\n".join(sources_md_parts) if sources_md_parts else "(no sources retrieved)"
        return answer, sources_md

    with gr.Blocks(title="Urdu RAG (Qwen 2.5 7B + Wikipedia-ur)") as demo:
        gr.Markdown("# Urdu RAG\nQwen 2.5 7B fine-tuned for Urdu + retrieval over Urdu Wikipedia.")
        with gr.Row():
            with gr.Column(scale=2):
                inp = gr.Textbox(label="Sawal (سوال) / Question", lines=2, rtl=False)
                btn = gr.Button("Pucho")
                out_answer = gr.Markdown(label="Jawab")
            with gr.Column(scale=1):
                out_sources = gr.Markdown(label="Wikipedia sources")
        btn.click(respond, inputs=[inp, gr.State([])], outputs=[out_answer, out_sources])
        inp.submit(respond, inputs=[inp, gr.State([])], outputs=[out_answer, out_sources])
        gr.Examples(
            examples=[
                "Pakistan ka sab se bara sooba rakbe ke lehaz se kaunsa hai?",
                "Quaid-e-Azam Muhammad Ali Jinnah ne Pakistan ki azadi mein kya kirdar ada kiya?",
                "Pakistan ka qaumi tarana kis ne likha?",
                "علامہ اقبال کا اصل نام کیا تھا؟",
            ],
            inputs=inp,
        )
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
