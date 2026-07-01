# v3: fixing the regressions, and what RAG actually bought me

*Build-in-public log, June 2026. Fine-tuning Qwen 2.5 7B for Urdu, on a hobby budget.*

## Where v2 left me

v2 was a real model — it beat the base Qwen about 66% of the time across three LLM judges, and code-mixed Urdu went from 0% to 80% after I added Roman-Urdu code data. But the eval also showed three categories had gone *backwards* from v1:

- summarization: 73% → 46%
- reasoning: 44% → 31%
- grammar: 40% → 36%

And separately, I'd spent a couple of weeks trying to bolt retrieval (RAG) onto v2, and it failed in an embarrassing way. The fine-tuned model had only ever seen plain question→answer pairs in training, so when I fed it `{retrieved context}\n\nسوال: {question}`, that shape was out-of-distribution. It either ignored the context or got confused by it — RAG-FT lost to plain-FT on 17 of 100 prompts. When I tried the base model with RAG instead, it was worse: it leaked Chinese characters on 45 of 100 answers, because Qwen falls back to its Chinese pre-training when you hand it unfamiliar Urdu context.

So v3 had two jobs: **fix the three regressions, and train the model on the actual RAG format so retrieval stops being out-of-distribution.**

## The data

Everything new was synthetic, generated with GPT-4o-mini, about 5,800 examples for \$0.63:

- **Grammar pairs.** GPT-4o-mini *corrupts* a trusted, human-written Urdu sentence with one labelled error. The gold answer is the original trusted sentence — never a model rewrite. That way the model's own (imperfect) Urdu can never poison the grammar target.
- **Summarization + reasoning.** For summaries, the prompt asks for an exact sentence count and I verify the count in Python (retry once, else discard). For reasoning, Python owns *all* the arithmetic — it generates the numbers and the answer, and the model only narrates the steps in Urdu. I then check that the computed digits actually appear in the output. The model never does math.
- **RAG triples.** Grounded on real Urdu Wikipedia chunks, formatted *byte-for-byte* like the retrieval endpoint serves them. Three kinds: a single gold chunk, a gold chunk plus two distractors, and "noise" (three unrelated chunks where the right move is to say the context doesn't help).

A quick detour: while building this I read FAIR's new "Agentic Self-Instruct" paper, which has an agent generate, test, and refine training data in a loop. It's a genuinely good idea — but it's built for reinforcement learning with verifiable rewards, and I'm doing plain supervised fine-tuning. I'd also already done its core moves a cheaper way (grounding, deterministic checks, a human review gate). So I wrote down *why* I was skipping it and moved on. Reading a paper and deciding it doesn't fit is also engineering.

## The spot-check earned its keep

Before spending money on training, I reviewed 200 of the generated examples by hand. Two systemic bugs turned up that my automated self-tests had completely missed:

1. **One reasoning template was 100% broken.** The "sum of three ages" problems were rendering the scenario but dropping the actual question — *"Ayesha is 33, Tariq is 5 older, Imran is 2 younger."* and then... nothing. My validator only checked that the right digits appeared, not that a question existed, so it waved them all through. Every single age-sum example in the sample failed.

2. **The RAG "noise" answers were confidently wrong.** For the noise examples, the gold answer said "not in the context, but generally…" and then asserted a fact from GPT-4o-mini's own knowledge. Unverified. I ran a full check across all 282 noise rows and ~18% asserted something wrong or unverifiable — concentrated, alarmingly, in religious questions (Urdu Wikipedia is religion-heavy, so the noise questions skewed that way, so the model confabulated exactly where a wrong confident answer does the most harm).

I fixed both at the generator level — Python now owns the question the same way it owns the arithmetic, and the noise answers just decline instead of fabricating — regenerated the affected slices, and verified them clean (0 missing questions, 0 confabulations). The lesson I keep relearning: a deterministic check beats an LLM judge for the things it can actually check, and a human still beats both for the things neither can.

## Training

68,997 examples (the 63k from v2 plus the ~5,700 new ones), QLoRA on Qwen 2.5 7B, two epochs, sequence length bumped to 4096 so the RAG context fits. Final train loss 0.5738. About \$5–6 of H100 time on Modal.

## Results

I judged with Claude and GPT-5.3 (Gemini's free tier capped out at 20 requests a day, so it sat this one out). Three comparisons.

**v3 vs the base model — a clean win.**

| | win rate vs base |
|---|---|
| **overall** | **79.5%** |
| creative | 100% |
| translation | 97% |
| summarization | **82%** (was 46% in v2) |
| grammar | **82%** (was 36%) |
| qa | 79% |
| code explanation | 75% |
| code-mixed | 70% |
| reasoning | **53%** (was 31%) |

All three regressions recovered, and overall jumped from v2's ~66% to 79.5%. On its own, that's the result I wanted.

**v3 vs v2, head to head — a rebalance, not an upgrade.** v3 only won 43% of direct matchups. It clearly won summarization, grammar, and code-mixed, but *lost* translation, code-explanation, reasoning, and qa. Pouring in RAG/grammar/summarization data traded away strength elsewhere. There's no free lunch in the data mix, and I should stop expecting one.

**RAG — the honest one.** This is where I was most wrong.

The structural fix worked completely: **0 of 100 RAG answers leaked Chinese** (v2's base+RAG was 45/100). And on the questions that need a fact, RAG *corrects* the model — asked for Pakistan's largest province by area, plain v3 says "Punjab" (wrong), and RAG says "Balochistan, 347,190 km²" straight out of the retrieved article.

But when I judged RAG-v3 against plain v3 across all 100 prompts, **RAG won only 15.5%** — basically the same as v2's 17%. My hypothesis that RAG-aware training would flip RAG into a net win was just wrong, and here's why: 79 of the 100 eval prompts are creative writing, grammar, reasoning, translation, code — things where retrieved Wikipedia context is pure noise. RAG should lose those (it scored 0% on creative and summarization, which is correct behaviour). Even on factual QA it only won 31%, because plain v3 often already knows the answer and phrases it more cleanly than a context-stitched RAG response. A preference judge rewards "fluent and right" over "grounded but clunky," so it under-credits the times RAG is more *accurate* but less smooth.

So RAG didn't become a blanket upgrade. What it became is *safe* — a deployable grounding tool you can point a factual query at without it collapsing into Chinese word-salad. v2 never had that. That's a smaller, truer claim than the one I started with.

## What I'd put on the wall

- **RAG-aware training fixes the breakage, not the win-rate.** Retrieval helps the minority of prompts that actually need a looked-up fact. Measuring it on a blanket eval buries that signal under all the prompts that don't.
- **Data mixes rebalance; they don't strictly improve.** v3 is better than v2 at the things I targeted and worse at some things I didn't. Worth knowing before you tell anyone "v3 > v2."
- **The cheap, boring quality gates caught the expensive bugs.** A Python digit-check and a human reading 200 rows found two systemic failures that a clean-looking automated test suite missed entirely.
- **A negative result you instrumented well is still a result.** I can say exactly *why* RAG doesn't win, which is more useful than a vague "it helped."

## Cost

v3 came in around \$9–10 (\$0.63 data, ~\$5–6 training, ~\$2–3 eval). The whole project — three model versions, two RAG attempts, hundreds of eval judgments — is still comfortably under the \$80 ceiling I set for myself.
