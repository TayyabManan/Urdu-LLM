# Fine-Tuning an Open LLM for Urdu

## A Thirty-Day Project Roadmap

## Muhammad Tayyab

```
MS Artificial Intelligence Engineering, COMSATS University Islamabad
tayyabmanan.com • linkedin.com/in/tayyabmanan
```
```
April 2026
```
## Abstract

```
Urdu is the national language of Pakistan and a first or second language for approximately
240 million people, yet it remains substantially under-served by openly available instruction-
tuned large language models. This document specifies a thirty-day roadmap for producing
a QLoRA-fine-tuned derivative of Qwen 2.5 7B Instruct, specialised on Urdu instruction-
following. The project follows a build-in-public methodology: all artefacts (training code, data
pipeline, merged model, adapter weights, and evaluation prompts) will be released openly, and
weekly progress will be documented publicly. The roadmap includes a five-day preparatory
phase to de-risk the fine-tuning pipeline before the public commitment window begins. Total
infrastructure cost is projected at US$30–70, with a single NVIDIA A100 rental accounting
for the majority of expenditure. The definition of done requires the fine-tuned model to
outperform the base model on at least 60% of a hand-curated Urdu evaluation set covering
question answering, summarisation, reasoning, and creative writing tasks.
```
## Contents

**1 Introduction and Motivation 2**
1.1 Author context..................................... 2
1.2 Positioning....................................... 3

**2 Project Scope and Definition of Done 3**
2.1 In scope......................................... 3
2.2 Out of scope...................................... 3
2.3 Definition of done.................................... 3

**3 Technical Architecture 4**
3.1 Component selection.................................. 4
3.2 Hyperparameter starting point............................ 4

**4 Data Strategy 5**
4.1 Target volume...................................... 5
4.2 Source composition................................... 5
4.3 Cleaning and validation................................ 5
4.4 Format.......................................... 5

**5 Evaluation Plan 6**
5.1 Evaluation prompt set................................. 6
5.2 Comparison methodology............................... 6


```
5.3 Success criterion.................................... 6
```
```
6 Timeline 6
6.1 Week 0: Preparation (Days−5 to−1)........................ 6
6.2 Week 1: Data pipeline................................. 7
6.3 Week 2: Full training.................................. 7
6.4 Week 3: Evaluation and iteration........................... 7
6.5 Week 4: Launch..................................... 8
```
```
7 Cost Estimate 8
```
```
8 Risk Register 8
```
```
9 Dissemination Plan 8
9.1 Posting schedule.................................... 8
9.2 Platforms........................................ 9
```
```
A Day 0 Announcement Post 9
```
```
B Weekly Post Templates 10
```
```
C References and Resources 11
```
## 1 Introduction and Motivation

```
Urdu occupies an unusual position in the current language-model landscape. It is a high-resource
language by demographic measure—first or second language for roughly 240 million speakers
across Pakistan and diaspora communities—yet a low-resource language by computational
measure. Commercial frontier models (GPT-4, Claude, Gemini) handle Urdu acceptably but
produce output that reads as translated rather than native. The open-model ecosystem offers
little: most Urdu capability in open 7B-scale models derives from incidental exposure during
multilingual pretraining, not from deliberate instruction-tuning.
This gap is the project’s motivation. The objective is not to advance research frontiers. It is to
apply a known recipe—QLoRA fine-tuning of a multilingual base model on a curated instruction
dataset—to a language that has not received adequate attention from the open community, and
to release the result openly.
```
**1.1 Author context**
The author is a first-year MS Artificial Intelligence Engineering candidate at COMSATS Univer-
sity Islamabad, with a BS in Geographic Information Systems from the University of the Punjab
(2021–2025). Professional experience includes two years as an AI Developer at Cointegration, with
hands-on work in multi-agent systems (LangChain, AutoGen, CrewAI), retrieval-augmented gen-
eration, and production LLM tooling. Relevant adjacent experience includes model deployment
to Hugging Face Hub and GPU-based workflows on Modal.
This will be the author’s first fine-tuning project end-to-end. Accordingly, the roadmap includes
a five-day preparatory phase during which the full training loop is exercised on a small surrogate
task before the public commitment begins.


**1.2 Positioning**

The project is scoped deliberately as a practitioner exercise rather than a research contribution.
The deliverable is evidence of engineering competence across the full fine-tuning lifecycle—data
curation, training, evaluation, deployment, and dissemination—not a novel methodological claim.
This framing is appropriate for a portfolio piece and aligns with the author’s career objective of
securing a full-time machine-learning engineer position.

## 2 Project Scope and Definition of Done

**2.1 In scope**

- QLoRA fine-tuning of a single 7–8B parameter open multilingual base model.
- Urdu instruction-tuning using a hybrid dataset (translated, native, and hand-crafted).
- Quantitative and qualitative evaluation against the base model on a hand-curated Urdu
    prompt set.
- Public release of LoRA adapter weights and merged model on Hugging Face Hub.
- A live Gradio demonstration on Hugging Face Spaces.
- A concluding write-up covering methodology, results, failure cases, and lessons learned.

**2.2 Out of scope**

- Pretraining from scratch or continued pretraining on raw Urdu corpora.
- Training beyond a single 7–8B parameter base model.
- Direct head-to-head comparison with closed commercial frontier models.
- Formal academic publication. A paper-style extension of this work is not precluded but is
    not a Day-30 deliverable.
- Coverage of languages other than Urdu (e.g. Punjabi, Sindhi, Pashto).

**2.3 Definition of done**

The project is considered complete when _all_ of the following conditions are satisfied:

```
1.The fine-tuned model (merged weights) and LoRA adapter are published to Hugging Face
Hub under an open license.
```
2. A Gradio demo is deployed on Hugging Face Spaces and is reachable at a public URL.

```
3.The GitHub repository contains a complete training pipeline, evaluation scripts, the curated
evaluation prompt set, and a reproducibility-oriented README.
```
4. A public write-up has been published covering methodology, results, and limitations.

```
5.On the author’s curated evaluation set of approximately 100 prompts, the fine-tuned model
is judged (by the author, with spot-checks by at least one other Urdu speaker) to produce a
preferable response on at least 60% of prompts relative to the base model.
```

## 3 Technical Architecture

```
3.1 Component selection
```
```
Component Choice Rationale
Base model Qwen 2.5 7B Instruct Strongest multilingual prior of any
open 7B-class model. Superior Urdu
coverage to Llama 3.1 8B and Mistral
7B.
Fine-tune method QLoRA (4-bit quantised LoRA) Full fine-tune is infeasible on a sin-
gle GPU. QLoRA reduces VRAM to
≈16 GB while preserving task perfor-
mance. Adapter is∼100 MB.
Training framework Unsloth Approximately 2× throughput over
stock Hugging Face TRL on consumer
GPUs; simpler configuration for first-
time users.
Compute Modal (A100 80GB) Author has prior experience
with Modal. Serverless pricing
(∼US$1.80/hr) suits bursty workloads.
Good Hugging Face Hub integration.
Experiment tracking Weights & Biases De facto standard; free for individual
use. Loss curves are necessary for de-
bugging and for public-facing content.
Model hosting Hugging Face Hub (adapter & merged) Author has prior experience; free; best
reach.
Demo hosting Hugging Face Spaces (Gradio) Free tier sufficient; deployment paral-
lels author’s existing HF workflow.
Version control GitHub (public repository) Standard. Repository structure follows
Hugging Face community conventions.
```
```
Table 1: Selected components and the justification for each choice.
```
**3.2 Hyperparameter starting point**
The following starting values are drawn from Unsloth’s Qwen 2.5 QLoRA recipe and the Aya
Collection training configurations. They are treated as a starting point, not a final specification;
Week 2 includes time for adjustment.

- LoRA rank ( _r_ ): 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Learning rate: 2 × 10 −^4 , cosine schedule, 3% warmup
- Batch size: 2 per device, gradient accumulation 8 (effective batch 16)
- Epochs: 2
- Max sequence length: 2048 tokens
- Optimiser: AdamW 8-bit


- Precision: bfloat16 forward, 4-bit NF4 base weights

## 4 Data Strategy

Data is the single largest source of risk and the largest share of calendar time in this roadmap.
Week 1 is substantially dedicated to data work.

```
4.1 Target volume
Target: approximately 50,000 instruction-response pairs in Urdu script (Nastaliq/Naskh), with
a smaller Roman Urdu component as a stretch goal if time permits.
```
```
4.2 Source composition
Three sources are combined, with the proportion tuned during Week 1:
```
1. **Translated English instruction data (bulk).** The Alpaca-cleaned dataset (~52k ex-
    amples) translated to Urdu via a capable LLM (e.g. Claude, GPT-4) or the Google Cloud
    Translation API. This provides volume, diversity of instruction types, and baseline coherence.
    The principal risk is translation artefacts leaking into the model’s style.
2. **Native Urdu instruction data (quality).** Cohere for AI’s Aya Dataset contains a
    non-trivial Urdu subset written by native speakers. This component is smaller in volume
    but substantially improves stylistic fidelity.
3. **Hand-crafted Pakistani-context examples (character).** Approximately 300–500 ex-
    amples written by the author covering topics that machine translation typically handles
    poorly: Pakistani current affairs framed in Urdu, Urdu grammar and morphology questions,
    regional idioms, and code-mixed (Urdu–English) queries that reflect how bilingual Pakista-
    nis actually interact with AI assistants. This component is the project’s signature and is
    disproportionately responsible for whether the model _feels_ Pakistani.

```
4.3 Cleaning and validation
```
- Deduplication on instruction-response pairs.
- Length filtering: examples outside the range 20–2000 tokens (either side) are dropped.
- Script detection: examples containing predominantly non-Urdu script are dropped unless
    they are intentionally code-mixed (handled via a reserved prefix).
- Spot-check: a 200-example random sample is manually reviewed for translation quality,
    tokenisation correctness, and semantic coherence before the full training run.

```
4.4 Format
All data is harmonised to a single Alpaca-style JSONL schema:
```
```
{"instruction": "...", "input": "...", "output": "..."}
```
```
The Qwen chat template is applied at training time, not at data-preparation time, to preserve
flexibility.
```

## 5 Evaluation Plan

Automated benchmarks for Urdu are weak. The project relies primarily on a hand-curated
evaluation prompt set, supplemented by any applicable subset of MMLU-Urdu (if available at
the time of evaluation).

**5.1 Evaluation prompt set**

Approximately 100 prompts, distributed across the following task categories:

- Open-domain question answering (∼20 prompts)
- Summarisation of short Urdu passages (∼15 prompts)
- Translation: Urdu to English, English to Urdu (∼15 prompts)
- Urdu grammar and morphology questions (∼10 prompts)
- Multi-step reasoning in Urdu (∼15 prompts)
- Creative writing: short Urdu composition tasks (∼10 prompts)
- Code explanation in Urdu (∼10 prompts)
- Code-mixed (Urdu–English) queries (∼5 prompts)

**5.2 Comparison methodology**

For each prompt, both the base Qwen 2.5 7B Instruct and the fine-tuned model generate a
response under identical decoding parameters (temperature 0.7, top- _p_ 0.9, max 512 tokens).
Responses are presented side-by-side in a blinded comparison spreadsheet and rated pairwise (A
better / B better / tie) by the author, with a subset validated by at least one other native Urdu
speaker.

**5.3 Success criterion**

A preference rate of≥60%for the fine-tuned model, across the evaluation set, satisfies the
success criterion. A preference rate between 50% and 60% indicates marginal improvement and
triggers a second training iteration. Below 50% indicates a regression and requires diagnosis
before publication.

## 6 Timeline

The project spans 35 calendar days: a 5-day preparatory phase (Week 0) followed by a 30-day
public commitment window (Weeks 1–4).

**6.1 Week 0: Preparation (Days** − **5 to** − **1)**

Purpose: to surface and resolve all infrastructure friction before the public commitment begins.
No public posts during this phase.

- **Day** − **5:** Create GitHub repository (private initially). Set up Modal environment. Install
    Unsloth in a Modal container and verify GPU access.
- **Day** − **4:** Run Unsloth’s official Llama 3.2 1B tutorial notebook end-to-end on a trivial
    dataset (500 examples). Confirm the full loop: data loading, training, adapter save, inference.


- **Day** − **3:** Repeat the tutorial on Qwen 2.5 0.5B with 1,000 examples drawn from a public
    instruction dataset, to surface any Qwen-specific issues at a small scale.
- **Day** − **2:** Set up Weights & Biases tracking. Push a throwaway adapter to Hugging Face
    Hub to confirm the upload flow works.
- **Day** − **1:** Repository structure finalised. README skeleton in place. Initial data-collection
    scripts drafted (Alpaca download, Aya filter). Day 0 post drafted and ready.

By the end of Week 0, the author has executed the full fine-tuning loop at least twice at small
scale. The public commitment begins from a position of demonstrated capability, not speculation.

**6.2 Week 1: Data pipeline**

- **Day 1:** Day 0 post published. Repository goes public. First commit includes the Week 0
    scripts and the roadmap.
- **Days 2–3:** Alpaca translation run (Urdu). This is done in batches with validation spot-checks,
    not in a single pass.
- **Days 4–5:** Aya Urdu subset pulled and filtered. Hand-crafted examples begin (target: 50
    per day).
- **Day 5:** Week 1 post published. Topic: the data pipeline, including honest discussion of what
    was thrown away and why.
- **Days 6–7:** Dry-run fine-tune on 5,000 examples to verify the pipeline end-to-end on the
    target base model. Loss curve inspected; no deliverables depend on this output.

**6.3 Week 2: Full training**

- **Days 8–9:** Final data cleaning; unified JSONL produced and checksummed.
- **Days 9–10:** Full QLoRA training run. Estimated wall-clock time: 6–10 hours on A
    80GB. Weights & Biases tracks loss, gradient norms, learning rate.
- **Day 11:** Sanity-check early generations. If early outputs are degenerate, diagnose (typical
    culprits: template mismatch, learning rate too high, data quality issue).
- **Days 12–13:** Merge LoRA adapter with base model, quantise for deployment, push both
    adapter and merged model to Hugging Face Hub (v0.1).
- **Day 14:** Week 2 post published. Topic: loss curves, first sample outputs (good and bad),
    what was surprising.

**6.4 Week 3: Evaluation and iteration**

- **Days 15–17:** Evaluation prompt set finalised (100 prompts, categorised). Both models run
    on the full set; outputs logged.
- **Day 18:** Blinded pairwise comparison conducted. Preference rate computed per category
    and overall.
- **Day 19:** Failure analysis. If preference rate _<_ 60% overall, or _<_ 40% in any single category,
    a second training iteration is planned.
- **Days 20–21:** Second iteration (if required): data rebalanced, hyperparameters tuned, retrain.


```
Otherwise, time is allocated to polishing.
```
- **Day 21:** Week 3 post published. Topic: evaluation results, including categories where the
    model _worsened_. Honesty is the point.

**6.5 Week 4: Launch**

- **Days 22–24:** Gradio demo built and deployed to Hugging Face Spaces. README completed.
    License (Apache 2.0 or similar permissive) selected.
- **Days 25–26:** Launch write-up drafted. Contents: the motivation, the stack, the data
    strategy, training details, evaluation results, failure cases, reproducibility instructions, next
    steps.
- **Day 27: Launch post.** The model, the demo URL, the write-up, and the repository are
    released together. Relevant communities and individuals are notified (Urdu NLP researchers,
    local ML communities, Hugging Face community tag).
- **Days 28–30:** Community response is monitored. At least one follow-up post addresses
    feedback or adds a small requested feature. Project is declared shipped.

## 7 Cost Estimate

**Item Cost (USD) Notes**

Modal A100 80GB rental 20–45 ∼15–25 hours aggregate across Week 0 tutorials, Week 1
dry run, Week 2 full training, and any Week 3 iteration.
Translation API (optional) 0–20 Zero if an already-available LLM is used for translation;
∼US$15–20 if a paid translation API is used.
Hugging Face Hub 0 Free tier is sufficient for adapter, merged model, and Spaces
demo.
Weights & Biases 0 Free for individual use.
GitHub 0 Free for public repositories.
Domain/hosting 0 The project uses existing author infrastructure (tayyab-
manan.com).

**Projected total 20–65** Realistic single-run estimate. A 40% buffer (∼US$80 ceil-
ing) is advisable to cover a second training iteration if
required.

```
Table 2: Projected infrastructure expenditure.
```
## 8 Risk Register

## 9 Dissemination Plan

The project follows a build-in-public methodology. Consistent cadence is prioritised over optimal
individual posts.

**9.1 Posting schedule**

- **Day 1:** Day 0 announcement post (template: Appendix A).
- **Day 5:** Week 1 post — data pipeline.


**Risk Severity Mitigation**

Training loop fails during execution High Week 0 is dedicated to de-risking this; author
will have run the loop twice at small scale before
committing publicly.
Translated Alpaca data is low quality High Spot-check 200 examples before full training;
blend with Aya (native) and hand-crafted exam-
ples; be willing to regenerate translation subset.
Fine-tune worsens the base model Medium Evaluation plan includes a measurable 60% prefer-
ence threshold. Week 3 reserves time for a second
iteration. Honest reporting is required regardless.
A100 availability on Modal Low Modal is generally responsive; fallback options
include L4 or smaller-batch configurations.
Public commitment anxiety Medium Week 0 produces internal confidence before the
Day 0 post. The Day 0 post is factual (“I am
doing X”), not promotional (“I will succeed at
X”).
Scope creep (multilingual, larger model) Medium Explicit out-of-scope list; any deviation is post-
poned to a post-project phase.
Evaluation bias (self-rating) Medium At least one additional native Urdu speaker vali-
dates a subset of the pairwise ratings.
Published model causes harm Low The release includes a model card describing limi-
tations. No safety-critical use cases are claimed.
Standard refusal behaviours from the base model
are preserved.

```
Table 3: Anticipated risks, severity, and mitigation strategies.
```
- **Day 14:** Week 2 post — training and first outputs.
- **Day 21:** Week 3 post — evaluation results, including regressions.
- **Day 27:** Launch post — model, demo, write-up, repository.
- **Day 30:** Retrospective post — what worked, what did not, what the author would do
    differently.

Additional micro-posts (loss curve screenshots, notable failure cases, interesting data finds) are
posted opportunistically but are not required by the plan.

**9.2 Platforms**

Primary: LinkedIn. Secondary: X (Twitter) cross-post. The GitHub repository and Hugging
Face model card serve as the canonical technical artefacts and are linked from every post.

## A Day 0 Announcement Post

The following is a paste-ready template for the Day 0 announcement. Bracketed phrases should
be reviewed but not materially altered.

```
I’m publicly fine-tuning an open LLM for Urdu in the next 30 days.
Here’s why it matters. Pakistan has roughly 240 million people. Most of us think, speak, and write in
```

```
Urdu or Roman Urdu every day. But almost every LLM we use was trained mostly in English, and
when we ask in Urdu, the outputs feel translated—not native. There isn’t a great open Urdu-specialised
model. That’s a gap worth closing.
So for the next 30 days I’m going to build one. Publicly. With the ugly parts included—loss curves
that don’t converge, datasets that turn out to be garbage, decisions I’ll probably regret. All of it.
Here’s the plan I’m starting with (it will change):
Base: Qwen 2.5 7B Instruct (strong multilingual prior).
Method: QLoRA fine-tune on a single A100 rental.
Data: Translated Alpaca, Aya Urdu subset, and ∼ 500 hand-crafted Pakistani-context examples.
Eval: hand-built Urdu prompt set covering QA, summarisation, reasoning, and creative writing.
End state: model on Hugging Face Hub, Gradio demo, full write-up.
I’ll post updates every few days. Repo goes public on Day 1. Hard commit: it ships in 30 days, no
matter how rough.
Follow along if this is your thing. And if you’ve fine-tuned for a low-resource language before, I would
genuinely appreciate anything you wish you’d known on day 0.
```
## B Weekly Post Templates

**Week 1 — Data pipeline**
_Week 1 of the Urdu LLM fine-tune: data.
I spent most of this week translating Alpaca to Urdu, pulling the Aya Urdu subset, and hand-writing
examples in Urdu covering [specific topics]. Here’s what I learned:
[Finding 1 about translation quality — with a concrete example of a bad translation and how it was
caught.]
[Finding 2 about the Aya subset — what it’s genuinely good at.]
[Finding 3 about hand-crafted examples — the category of questions where translation consistently
fails.]
Final dataset: [N] examples, split [percentages] across the three sources. Dry-run training on a 5k
subset looks healthy. Full training run starts this week.
Repo: [link]._

**Week 2 — Training and first outputs**
_Week 2: training done. Early outputs below.
[Loss curve screenshot.] The dip at step [X] is where [event]; flat section from [Y] is [reason].
First sample outputs (base Qwen vs fine-tune):
[3–5 illustrative examples — mix of wins, losses, and surprises.]
What went well: [something specific].
What didn’t: [something specific].
Model is at Hugging Face Hub: [link]. Evaluation starts next week._

**Week 3 — Evaluation**


```
Week 3: evaluation results. The headline number is [X%] preference for the fine-tune over the base
across [N] prompts.
Where it won: [categories where the fine-tune was strongest].
Where it lost or tied: [categories where improvement was marginal or negative]. The clearest regression
was [specific example]. My best guess at the cause is [hypothesis].
Based on this, I am [deciding to iterate OR declaring v0.1 final]. Launch post on [date].
```
```
Week 4 — Launch
I just released [model name] — a fine-tuned Urdu version of Qwen 2.5 7B, built over the last 30 days.
Model: [HF link]. Demo: [HF Spaces link]. Code: [GitHub link]. Write-up: [blog post link].
Results summary: [one-sentence performance claim] on [evaluation set description]. Full methodology,
failure cases, and limitations are in the write-up.
This is v0.1. Feedback—especially from Urdu speakers—is what determines what v0.2 looks like. Try
the demo. Tell me what it gets wrong.
Thanks to everyone who commented with suggestions during the build. Credits in the README.
```
## C References and Resources

```
Qwen 2.5 Technical Report & model card:https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```
```
Unsloth repository and documentation:https://github.com/unslothai/unsloth
```
```
Cohere for AI Aya Dataset:https://huggingface.co/datasets/CohereForAI/aya_dataset
```
```
Alpaca-cleaned dataset:https://huggingface.co/datasets/yahma/alpaca-cleaned
```
```
Modal documentation:https://modal.com/docs
```
Weights & Biases: https://wandb.ai

```
PEFT library (QLoRA):https://huggingface.co/docs/peft
```
```
Hugging Face TRL (SFTTrainer):https://huggingface.co/docs/trl
```
```
— End of Roadmap —
```

