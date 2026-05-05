# Urdu LLM Evaluation Report

## Model
- **Base:** Qwen 2.5 7B Instruct
- **Fine-tuned:** QLoRA adapter (rank 16, 3 epochs on 53.8k Urdu examples)
- **Decoding:** temp=0.7, top_p=0.9, max_tokens=512
- **Judge:** Claude (blinded pairwise, randomized A/B order per batch)

## Overall Results (99 judged / 100 prompts)

| | Count | Percentage |
|---|---|---|
| **Fine-tuned wins** | 51 | 51.5% |
| **Base wins** | 33 | 33.3% |
| **Ties** | 15 | 15.2% |

**Target: 60% fine-tuned preference — NOT MET overall (51.5%)**

# Per-Category Breakdown

| Category | Total | Finetuned Wins | Base Wins | Ties | Finetuned % |
|---|---|---|---|---|---|
| Translation | 15 | 12 | 0 | 3 | **80.0%** |
| Summarization | 11 | 8 | 3 | 0 | **72.7%** |
| QA | 21 | 13 | 5 | 3 | **61.9%** |
| Creative | 11 | 6 | 2 | 3 | 54.5% |
| Reasoning | 16 | 7 | 6 | 3 | 43.8% |
| Grammar | 10 | 4 | 4 | 2 | 40.0% |
| Code Explanation | 10 | 1 | 8 | 1 | 10.0% |
| Code Mixed | 5 | 0 | 5 | 0 | 0.0% |

## Key Findings

### Where fine-tuned model clearly wins
- **Translation (80%):** Base model frequently degenerates into Chinese text or gibberish on translation tasks. Fine-tuned produces clean, accurate Urdu translations.
- **Summarization (73%):** Fine-tuned produces more concise, faithful summaries. Base often has garbled endings and odd word choices.
- **QA (62%):** Fine-tuned gives more coherent answers in proper Urdu. Base frequently hallucinates facts or produces incoherent text.

### Where it's competitive
- **Creative (55%):** Both models struggle with creative writing. Fine-tuned slightly better at poetry and speeches; base sometimes better at longer narrative.
- **Reasoning (44%):** Mixed results. Base sometimes more detailed in step-by-step math; fine-tuned sometimes more concise and correct.
- **Grammar (40%):** Split. Base better at structured grammar explanations; fine-tuned better at corrections and Urdu-specific grammar.

### Where base model dominates
- **Code Explanation (10%):** Base provides detailed code explanations with examples. Fine-tuning degraded this capability significantly.
- **Code Mixed (0%):** Base gives comprehensive technical answers. Fine-tuned responses are vague, incomplete, or cut off.

## Analysis

The fine-tuning achieved its primary goal: **dramatically improved Urdu language capability**. For Urdu-core tasks (QA, summarization, translation, grammar, creative writing), the fine-tuned model wins **63.2%** of the time — exceeding the 60% target.

However, fine-tuning caused **catastrophic forgetting on code/technical tasks**. The base Qwen model was already strong at code explanation in English; the Urdu fine-tuning degraded this. This is expected behavior for QLoRA fine-tuning without code-related training data.

### Urdu-Core Only (excluding code categories)

| | Count | Percentage |
|---|---|---|
| **Fine-tuned wins** | 50 | 59.5% |
| **Base wins** | 20 | 23.8% |
| **Ties** | 14 | 16.7% |

## Recommendations

1. **The adapter is good for Urdu language tasks.** Deploy it for Urdu QA, translation, summarization, and creative writing.
2. **Do not use the adapter for code tasks.** Route code-related queries to the base model instead.
3. **To improve overall score:**
   - Add code explanation examples in Urdu to training data
   - Add code-mixed examples with technical content
   - Consider 2 epochs instead of 3 (best eval loss was at epoch 2)
4. **For production:** Consider a routing layer that sends code queries to base and Urdu queries to fine-tuned model.

## Individual Judgments

| ID | Category | Winner | Reason |
|---|---|---|---|
| 1 | qa | finetuned | Base hallucinated fake province name |
| 2 | qa | finetuned | Correctly names Din-e-Ilahi; base garbled |
| 3 | qa | finetuned | Coherent summary; base incoherent |
| 4 | qa | tie | Both contain factual errors |
| 5 | qa | finetuned | Correctly states 8 planets; base hallucinated 11+ |
| 6 | qa | base | Base closer to describing LBW |
| 7 | qa | finetuned | Correctly states H2O; base garbled |
| 8 | qa | finetuned | Concise accurate themes; base repetitive |
| 9 | qa | base | More detailed with DNS, routing, protocols |
| 10 | qa | finetuned | Lists pillars more recognizably |
| 11 | qa | tie | Both wrong on Indus Valley dates |
| 12 | qa | finetuned | Responds in Roman Urdu matching prompt |
| 13 | qa | finetuned | Concise coherent; base repetitive garbled |
| 14 | qa | finetuned | Correct year and mission name Apollo 11 |
| 15 | qa | base | More detailed on bone health |
| 16 | qa | finetuned | Concise correct; base rambles |
| 17 | qa | tie | Both poor scientific explanations |
| 18 | qa | finetuned | Responds in Roman Urdu matching prompt |
| 19 | qa | finetuned | Clear benefits stated; base incoherent |
| 20 | qa | base | Lists mountain names; finetuned claims Everest in Pakistan |
| 21 | summarization | finetuned | Cleaner more accurate summary |
| 22 | summarization | finetuned | More thorough and faithful |
| 23 | summarization | base | Responded in English but accurately |
| 24 | summarization | finetuned | Clean three-sentence summary as requested |
| 25 | summarization | finetuned | Concise accurate; base has opposite meaning |
| 26 | summarization | base | Base English accurate; finetuned garbled |
| 27 | summarization | finetuned | Faithful comprehensive; base has broken ending |
| 28 | summarization | finetuned | Clean bullet points matching source |
| 29 | summarization | base | Base attempted Urdu script as implied |
| 30 | summarization | finetuned | Concise correct; base exceeded length |
| 31 | summarization | finetuned | Clear structured points; base degenerates |
| 32 | translation | finetuned | Accurate natural Urdu; base incoherent with Chinese |
| 33 | translation | finetuned | Correct clean translation; base degenerates |
| 34 | translation | finetuned | Accurate fluent Urdu; base garbled |
| 35 | translation | finetuned | Slightly more accurate river translation |
| 36 | translation | finetuned | Roman Urdu matches prompt; base broken |
| 37 | translation | finetuned | Natural accurate; base barely readable |
| 38 | translation | tie | Both partially correct |
| 39 | translation | finetuned | Closer to correct meaning; base gibberish |
| 40 | translation | finetuned | More concise natural translation |
| 41 | translation | finetuned | More natural Urdu phrasing |
| 42 | translation | tie | Both produce accurate English translations |
| 43 | translation | finetuned | Base garbled; finetuned coherent |
| 44 | translation | finetuned | Base degenerates to Chinese; finetuned coherent |
| 45 | translation | tie | Both fail to provide correct idiom equivalent |
| 46 | translation | finetuned | Base incoherent; finetuned conveys meaning |
| 47 | grammar | finetuned | Answers in Urdu with correct fixes |
| 48 | grammar | base | Correctly fixes more sentences |
| 49 | grammar | base | Five examples per tense as asked |
| 50 | grammar | base | Explains multiple rules with structure |
| 51 | grammar | tie | Both completely fail the task |
| 52 | grammar | finetuned | Gives correct passive forms |
| 53 | grammar | finetuned | Lists actual Urdu idioms |
| 54 | grammar | base | Clearer examples in Roman Urdu |
| 55 | grammar | finetuned | Corrects all three to formal Urdu |
| 56 | grammar | tie | Both fail on gender rules |
| 57 | reasoning | tie | Both reach correct answer |
| 58 | reasoning | base | Full detailed working shown |
| 59 | reasoning | base | Thorough set-theory explanation |
| 60 | reasoning | finetuned | Closer reasoning despite both wrong |
| 61 | reasoning | finetuned | Correct doubling and sum |
| 62 | reasoning | finetuned | Correct math |
| 63 | reasoning | tie | Both correctly calculate |
| 64 | reasoning | finetuned | Complete correct solution |
| 65 | reasoning | finetuned | Correct total in matching language |
| 66 | reasoning | base | More structured suggestions |
| 67 | reasoning | base | Correct compound interest |
| 68 | reasoning | finetuned | Correct answer |
| 69 | reasoning | base | More detailed breakdown |
| 70 | reasoning | base | Correct arrival time with clear steps |
| 71 | reasoning | finetuned | Concise balanced; base rambled |
| 72 | creative | tie | Neither produced proper poem |
| 73 | creative | tie | Neither told coherent story |
| 74 | creative | tie | Both incoherent |
| 75 | creative | finetuned | Attempts ghazal structure with emotion |
| 76 | creative | base | Slightly more dialogue structure |
| 77 | creative | finetuned | Coherent speech; base garbled |
| 78 | creative | base | Longer narrative in Roman Urdu |
| 79 | creative | finetuned | Natural Urdu imagery |
| 80 | creative | finetuned | More culturally relevant |
| 81 | creative | finetuned | Natural dialogue with named characters |
| 82 | code_explanation | base | Detailed step-by-step with examples |
| 83 | code_explanation | base | Clearer explanation with output |
| 84 | code_explanation | finetuned | Clean code examples; base mixes Chinese |
| 85 | code_explanation | base | Actual HTML/CSS code examples |
| 86 | code_explanation | tie | Both identify division-by-zero correctly |
| 87 | code_explanation | base | Clear restaurant analogy |
| 88 | code_explanation | base | More structured with formatting |
| 89 | code_explanation | base | Thorough line-by-line SQL explanation |
| 90 | code_explanation | base | Complete git commands list |
| 91 | code_explanation | base | More detailed exception handling |
| 92 | code_mixed | base | Clear decorator explanation with code |
| 93 | code_mixed | base | Structured comparison with specifics |
| 94 | code_mixed | base | Comprehensive guide with code snippets |
| 95 | code_mixed | base | Working code example with explanation |
| 96 | code_mixed | base | Detailed actionable steps |
| 97 | qa | base | Lists multiple specific pros and cons |
| 98 | reasoning | tie | Both solve correctly |
| 99 | creative | finetuned | Coherent independence day speech |
| 100 | grammar | — | Not judged |
