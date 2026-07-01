"""HuggingFace Spaces frontend for the Urdu LLM v3 (Qwen 2.5 7B + LoRA).

Free-tier Spaces (no GPU) can't host a 7B model, so this app is a thin Gradio
proxy that POSTs to a Modal endpoint where the actual model lives. The Modal
endpoint URL is read from the Spaces Secret FT_ENDPOINT_URL.

Spaces secret to set:
    FT_ENDPOINT_URL = https://digitization--urdu-rag-ft-endpoint-ftserver-generate.modal.run
"""
import os
import gradio as gr
import httpx

ENDPOINT = os.environ.get("FT_ENDPOINT_URL")
TIMEOUT_S = 300  # Modal cold-starts up to ~3 min on first call
DEFAULT_SYSTEM = "You are a helpful assistant."

MODEL_TITLE = "Qwen 2.5 7B Urdu (v3 LoRA)"
HF_MODEL_URL = "https://huggingface.co/TayyabManan/qwen2.5-7b-urdu-v3"
GH_REPO_URL = "https://github.com/TayyabManan/Urdu-LLM"

INTRO = f"""# {MODEL_TITLE}

Fine-tuned Qwen 2.5 7B Instruct for Urdu (Urdu script + Roman Urdu + code-mixed).
**79.5% win rate vs base Qwen across 2 LLM judges** on a 100-prompt hand-curated set.

This Space is a Gradio frontend; the model runs on a private GPU endpoint (Modal H100).
First call may take 30-90s while the container warms up. Subsequent calls are 2-8s.

Model card · {HF_MODEL_URL}  |  Code & data · {GH_REPO_URL}
"""

EXAMPLES = [
    ["پاکستان کا قومی ترانہ کس نے لکھا اور کب لکھا گیا؟", 0.7, 512],
    ["علامہ اقبال کی شاعری میں 'خودی' کا تصور کیا ہے؟ مختصر بیان کریں۔", 0.7, 512],
    ["Mujhe Python mein decorators ka concept Roman Urdu mein samjhao.", 0.6, 512],
    ["درج ذیل انگریزی جملے کا اردو میں ترجمہ کریں: 'The weather in Lahore is pleasant in winter.'", 0.3, 256],
    ["لاہور کی پرانی گلیوں پر ایک مختصر نظم لکھیں۔", 0.9, 512],
]


def call_endpoint(prompt: str, system: str, max_tokens: int, temperature: float):
    """Generator: yields progress updates then final answer.

    Yielding lets the Markdown output update IMMEDIATELY when the user clicks
    Generate, rather than sitting empty for the 2-8s (warm) or 3-5min (cold)
    request duration. Gradio 5.x renders each yield in real time.
    """
    if not ENDPOINT:
        yield ("⚠️ **Server config missing.** `FT_ENDPOINT_URL` is not set in the "
               "Space secrets. The owner needs to add it under "
               "*Settings → Variables and secrets*.")
        return
    if not prompt or not prompt.strip():
        yield "_(empty prompt — type a question above and click Generate)_"
        return

    # 1. Immediate ack so the user knows the click registered
    yield ("⏳ **Sending request to the GPU endpoint…**\n\n"
           "_If this is the first call after a few minutes idle, the container "
           "is cold-starting (loads Qwen 7B + LoRA + embedder + reranker + BM25). "
           "Expect 30s–3min. Warm calls return in 2–8s._")

    payload = {
        "prompt": prompt.strip(),
        "system": system or DEFAULT_SYSTEM,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }
    try:
        import time
        t0 = time.time()
        with httpx.Client(timeout=TIMEOUT_S, follow_redirects=True) as http:
            r = http.post(ENDPOINT, json=payload)
            r.raise_for_status()
            data = r.json()
        elapsed = time.time() - t0
        text = data.get("text") or "_(empty response from endpoint)_"
        tokens_out = data.get("tokens_out")
        meta = f"\n\n---\n_↳ {elapsed:.1f}s"
        if tokens_out:
            meta += f" · {tokens_out} tokens out"
        meta += " · model: Qwen 2.5 7B + v3 LoRA · top_p 0.9, rep_penalty 1.1_"
        yield text + meta
    except httpx.TimeoutException:
        yield ("⚠️ **Endpoint timed out** after 5 minutes. The GPU container "
               "may still be cold-starting. Try again in 30 seconds — second "
               "call should be warm.")
    except httpx.HTTPStatusError as e:
        yield f"⚠️ **Endpoint error {e.response.status_code}**\n```\n{e.response.text[:300]}\n```"
    except Exception as e:
        yield f"⚠️ **Unexpected error**: `{type(e).__name__}: {e!s}`"


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;500;700&display=swap');

/* Base for input/output containers — moderate line-height that works for either script
   (the SCRIPT_DETECT_JS observer below tags individual elements with lang=ur|en
   for per-paragraph script-aware styling, see rules further down). */
.urdu-io, .urdu-io textarea, .urdu-io input {
    font-family: 'Noto Nastaliq Urdu', 'Noto Naskh Arabic', system-ui,
                 -apple-system, 'Segoe UI', sans-serif !important;
    font-size: 1.0rem;
    line-height: 1.75;
    unicode-bidi: plaintext;
    text-align: start;
}

/* Input textarea: rtl base (matches placeholder), but each line auto-flips by first char */
.urdu-io textarea {
    direction: rtl;
}

/* === Script-aware per-element styling (JS-tagged via MutationObserver) === */
/* Urdu/Arabic — Nastaliq w/ tall leading and right-aligned RTL */
.urdu-io [lang="ur"], .urdu-io [lang="ar"], .urdu-io [data-script="arabic"] {
    font-family: 'Noto Nastaliq Urdu', 'Noto Naskh Arabic', 'Jameel Noori Nastaleeq',
                 serif !important;
    font-size: 1.1rem;
    line-height: 2.15;
    direction: rtl;
    text-align: right;
}

/* English / Roman Urdu — clean sans-serif w/ normal leading and LTR */
.urdu-io [lang="en"], .urdu-io [data-script="latin"] {
    font-family: system-ui, -apple-system, 'Segoe UI', 'Helvetica Neue',
                 sans-serif !important;
    font-size: 1.0rem;
    line-height: 1.55;
    direction: ltr;
    text-align: left;
}

/* Examples row stays compact */
.gradio-container .examples { font-size: 0.95rem; }

footer { visibility: hidden }
"""

SCRIPT_DETECT_JS = """
() => {
    // Arabic-script Unicode ranges used by Urdu (+ Arabic supplements + presentation forms)
    const arabicRe = /[\\u0600-\\u06FF\\u0750-\\u077F\\u0870-\\u089F\\u08A0-\\u08FF\\uFB50-\\uFDFF\\uFE70-\\uFEFF]/;
    const latinRe = /[A-Za-z]/;

    function tagScript(root) {
        const blocks = root.querySelectorAll(
            '.urdu-io p, .urdu-io li, .urdu-io h1, .urdu-io h2, .urdu-io h3, .urdu-io blockquote'
        );
        blocks.forEach(el => {
            const txt = (el.textContent || '').trim();
            if (!txt) return;
            const arabicCount = (txt.match(/[\\u0600-\\u06FF\\u0750-\\u077F\\u0870-\\u089F\\u08A0-\\u08FF\\uFB50-\\uFDFF\\uFE70-\\uFEFF]/g) || []).length;
            const latinCount = (txt.match(/[A-Za-z]/g) || []).length;
            // Tag by dominant script — ties + no-script (numbers/punct only) default to current
            if (arabicCount > latinCount && arabicCount > 0) {
                el.setAttribute('lang', 'ur');
                el.setAttribute('dir', 'rtl');
            } else if (latinCount > 0) {
                el.setAttribute('lang', 'en');
                el.setAttribute('dir', 'ltr');
            }
        });
    }

    // Initial pass (catches static content)
    tagScript(document);

    // Re-tag whenever Gradio updates the DOM (streaming output, etc.)
    const observer = new MutationObserver((mutations) => {
        let needs_retag = false;
        for (const m of mutations) {
            if (m.type === 'childList' || m.type === 'characterData') {
                needs_retag = true;
                break;
            }
        }
        if (needs_retag) {
            // Debounce via rAF to avoid layout thrash
            requestAnimationFrame(() => tagScript(document));
        }
    });
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        characterData: true,
    });
}
"""

with gr.Blocks(
    title=MODEL_TITLE,
    theme=gr.themes.Soft(primary_hue="green"),
    css=CUSTOM_CSS,
    js=SCRIPT_DETECT_JS,
) as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Sawal (سوال) — Urdu, Roman Urdu, or English",
                placeholder="اپنا سوال یہاں لکھیں / Likhe apna sawaal yahan",
                lines=3,
                elem_classes=["urdu-io"],
            )
        with gr.Column(scale=1, min_width=180):
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05,
                                    label="Temperature",
                                    info="Lower = more factual / shorter")
            max_tokens = gr.Slider(64, 1024, value=512, step=64,
                                   label="Max new tokens")

    submit = gr.Button("Pucho / Generate", variant="primary")
    output = gr.Markdown(label="Jawab",
                         value="_(awaiting query)_",
                         line_breaks=True,
                         elem_classes=["urdu-io"])

    with gr.Accordion("Advanced: system prompt", open=False):
        system = gr.Textbox(label="System prompt",
                            value=DEFAULT_SYSTEM,
                            info=("Trained on \"You are a helpful assistant.\" — "
                                  "changing this may destabilize generation."),
                            lines=2)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt, temperature, max_tokens],
        label="Try these (factual QA, creative writing, code explanation, translation)",
    )

    inputs = [prompt, system, max_tokens, temperature]
    submit.click(call_endpoint, inputs=inputs, outputs=output, show_progress="full")
    prompt.submit(call_endpoint, inputs=inputs, outputs=output, show_progress="full")

    gr.Markdown(
        "---\n"
        f"_Built in public by [Muhammad Tayyab](https://linkedin.com/in/tayyabmanan). "
        f"Apache-2.0. Code, eval data, and training pipeline at [{GH_REPO_URL}]({GH_REPO_URL})._"
    )


if __name__ == "__main__":
    demo.launch()
