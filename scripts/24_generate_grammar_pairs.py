"""
Generate Urdu grammar-correction pairs so v3 recovers the grammar category.

The problem (from v2 eval): grammar correction regressed to 36% win rate. v2's
data had almost no "here is a wrong sentence, fix it" supervision, so the model
never reliably learned to repair agreement / tense / postposition errors.

The anti-poisoning trick (this is the whole point of the design):
    gpt-4o-mini's own Urdu is not trustworthy enough to be the GOLD answer of a
    grammar task — if we let it write the "correct" sentence we would be teaching
    the model the LLM's mistakes. So we never ask it to correct anything.

    Instead:
      1. We start from a TRUSTED, already-correct Urdu sentence S, sampled from
         data/cleaned/train.jsonl (human-written urdu-instruct / aya outputs that
         already passed the project's Urdu-ratio cleaning).
      2. We ask gpt-4o-mini to CORRUPT S — inject exactly ONE grammatical error,
         changing as little as possible — and label the error type.
      3. The training example is: instruction = "fix this sentence: <corrupted>",
         output = S (the trusted original, verbatim).

    The model that gpt-4o-mini produces is the WRONG sentence (errors are fine
    there — that's the input). The gold output is always the trusted human text.
    gpt-4o-mini never authors a gold answer, so it can't poison the category.

Per the user's choice, the output is the corrected sentence ONLY (no appended
explanation) — keeps the gold 100% trusted, no LLM-written note that could itself
be imperfect.

~25% of examples are Roman-Urdu (seeds from redgerd-roman / urdu-instruct-roman).
Roman has no orthographic gold, so for Roman seeds we only inject STRUCTURAL
errors (agreement / tense / postposition), never spelling or izafat.

Every corrupted sentence AND its trusted seed are screened against the 100 eval
prompts (src.data.eval_guard) so the grammar test set can't leak into training.

Run with:
    python scripts/24_generate_grammar_pairs.py --n 2000 --budget 1.50
Resume:
    python scripts/24_generate_grammar_pairs.py --n 2000 --budget 1.50 --resume
Smoke (tiny spend):
    python scripts/24_generate_grammar_pairs.py --n 6 --budget 0.10
Self-test (free, no API):
    python scripts/24_generate_grammar_pairs.py --self-test
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Local modules — sentence splitter + eval guard.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.chunking import split_sentences
from src.data.eval_guard import load_eval_guard, _normalize

load_dotenv()

# ─── Config ───
MODEL = "gpt-4o-mini"
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60
BATCH_SIZE = 8            # seeds per API call — corruption is cheap, batch big
SLEEP_BETWEEN = 0.5
TEMPERATURE = 0.7         # corruption needs error variety, not determinism

SEED_FILE = Path("data/cleaned/train.jsonl")
URDU_SOURCES = ("urdu-instruct", "aya-urdu")
ROMAN_SOURCES = ("redgerd-roman", "urdu-instruct-roman")

MIN_SEED_CHARS = 30
MAX_SEED_CHARS = 300
MIN_SEED_WORDS = 4        # need enough words to carry an agreement error
MAX_SEED_WORDS = 40

# Corruption-validity bounds (reject rewrites disguised as a single error).
LEN_RATIO_LO = 0.6
LEN_RATIO_HI = 1.6
EDIT_DIST_LO = 1         # at least one word changed
EDIT_DIST_HI = 4         # at most a few words — a single error, not a rewrite

OUTPUT_DIR = Path("data/grammar_pairs")
OUTPUT_FILE = OUTPUT_DIR / "grammar_pairs.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Error taxonomy. Spelling + izafat are Urdu-script-only (no Roman orthographic
# gold), so they're dropped for the Roman slice in the system prompt.
ERROR_TYPES_URDU = [
    "gender_agreement", "number_agreement", "verb_tense",
    "izafat", "postposition", "honorific", "spelling",
]
ERROR_TYPES_ROMAN = [
    "gender_agreement", "number_agreement", "verb_tense",
    "postposition", "honorific",
]

# Correction-request phrasings. NB: the user chose corrected-sentence-only output,
# so these only ask to CORRECT (no "identify the error" phrasings that would imply
# an explanation in the answer). Kept short so no phrasing alone forms a 5-gram
# match against an eval grammar prompt.
PHRASINGS_URDU = [
    "اس جملے کی گرامر درست کریں:",
    "درج ذیل جملے کو درست کریں:",
    "اس جملے کو درست اردو میں لکھیں:",
    "درج ذیل جملے کی تصحیح کریں:",
    "اس جملے کی گرامر کی غلطی ٹھیک کریں:",
    "نیچے دیے گئے جملے کو شستہ اردو میں لکھیں:",
]
PHRASINGS_ROMAN = [
    "Is jumlay ki grammar theek karo:",
    "Neeche diye gaye jumlay ko correct karo:",
    "Is jumlay ko sahi Roman Urdu mein likho:",
    "Is jumlay ki grammatical ghalti durust karo:",
]

SYSTEM_PROMPT = (
    "You corrupt grammatically-correct sentences to build grammar-correction "
    "training data. You receive a JSON array of already-correct sentences; each "
    "has an id, the sentence, and its script (\"urdu\" = Urdu/Nastaliq script, "
    "\"roman\" = Roman-Urdu in Latin letters).\n"
    "For EACH sentence:\n"
    "  1. Inject EXACTLY ONE grammatical error to make a corrupted version. "
    "Change as little as possible — ideally one to three words. Do NOT rephrase, "
    "translate, expand, shorten, or fix anything else. Keep the same words, "
    "meaning, length, and SCRIPT as the original except for the one error.\n"
    "  2. Pick the error type ONLY from that item's allowed_error_types list and "
    "report it as error_type. The types mean: gender_agreement (تذکیر/تانیث), "
    "number_agreement (واحد/جمع), verb_tense (فعل کا زمانہ), izafat (اضافت), "
    "postposition (حرفِ جار: نے/کو/سے/میں/پر/کا/کی/کے), honorific (آپ/تم + ہیں/ہے), "
    "spelling (common Urdu letter confusion). NEVER use a type outside the item's "
    "allowed_error_types (e.g. Roman items have no spelling or izafat).\n"
    "NEVER output a corrected sentence and NEVER add commentary. The corrupted "
    "sentence MUST stay in the same script as the input.\n"
    'Return ONLY JSON: {"items":[{"id":<int>,"corrupted":"<sentence with one '
    'error>","error_type":"<key>"}]}'
)

# ─── Script detection ───
URDU_CHARS = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
ASCII_LETTERS = re.compile(r"[A-Za-z]")


def script_ratios(text: str) -> tuple[float, float]:
    """Return (urdu_ratio, latin_ratio) over alphabetic characters."""
    u = len(URDU_CHARS.findall(text))
    a = len(ASCII_LETTERS.findall(text))
    tot = u + a
    if tot == 0:
        return 0.0, 0.0
    return u / tot, a / tot


def in_script(text: str, script: str) -> bool:
    u, a = script_ratios(text)
    if script == "urdu":
        return u >= 0.6
    return a >= 0.6 and u <= 0.1  # roman


# High-precision Roman-Urdu function words (token-level, lowercase). English and
# junk sentences in the Roman seed sources (transliterated translation answers,
# code fragments) carry almost none of these, so requiring >=2 hits drops them.
# Deliberately excludes tokens that are also English words ("is", "us", "the").
ROMAN_MARKERS = {
    "hai", "hain", "ho", "hota", "hoti", "hote", "raha", "rahi", "rahe",
    "tha", "thi", "ka", "ki", "ke", "ko", "mein", "se", "par", "pe",
    "ye", "yeh", "woh", "wo", "kya", "kyun", "kyunke", "kyunki",
    "nahi", "nahin", "aur", "ek", "hum", "tum", "aap",
    "kar", "karna", "karne", "karta", "karti", "karte",
    "jata", "jati", "jate", "gaya", "gayi", "gaye", "hua", "hui", "hue",
    "liye", "wala", "wali", "wale", "bohat", "bahut", "sab", "kuch",
    "apni", "apna", "apne", "mera", "meri",
}


def roman_urdu_score(text: str) -> int:
    """How many Roman-Urdu marker words appear (token-level, lowercase)."""
    return sum(1 for t in re.findall(r"[a-z]+", text.lower()) if t in ROMAN_MARKERS)


# urdu-instruct has task outputs that are English prose or labeled traces
# (sentiment labels, translations, "Reasoning:" / "Answer:" steps) — bad grammar
# seeds. META words are unambiguous junk markers (one is enough to reject). FUNC
# words are ordinary English function words (need >=2). Both exclude tokens that
# double as Roman-Urdu. Real Urdu / Roman-Urdu sentences carry neither.
ENGLISH_META = {
    "sentiment", "statement", "category", "positive", "negative", "translate",
    "translation", "following", "reasoning", "answer", "question", "example",
    "conveys", "sentence", "paragraph", "passage", "context", "explanation",
}
ENGLISH_FUNC = {
    "the", "of", "is", "are", "was", "were", "because", "which", "that",
    "this", "with", "its", "and", "for", "from",
}


def english_junk(text: str) -> bool:
    """True if the text reads as English meta-output rather than (Roman-)Urdu."""
    toks = re.findall(r"[a-z]+", text.lower())
    meta = sum(1 for t in toks if t in ENGLISH_META)
    func = sum(1 for t in toks if t in ENGLISH_FUNC)
    return meta >= 1 or func >= 2


def is_valid_seed(sent: str, script: str) -> bool:
    """Cheap, no-API checks that a candidate sentence is a usable seed."""
    sent = sent.strip()
    if not (MIN_SEED_CHARS <= len(sent) <= MAX_SEED_CHARS):
        return False
    if "\n" in sent or "**" in sent:
        return False  # structured / multi-part fragment or markdown artifact
    n_words = len(sent.split())
    if not (MIN_SEED_WORDS <= n_words <= MAX_SEED_WORDS):
        return False
    if english_junk(sent):
        return False  # English meta-output (sentiment / translation / reasoning)
    if not in_script(sent, script):
        return False
    # Roman sources are polluted with English (transliterated translation
    # answers) and code fragments; require Roman-Urdu markers to keep only real
    # Roman-Urdu. The Urdu-script slice can't have this problem — its script
    # ratio check already excludes Latin text.
    if script == "roman" and roman_urdu_score(sent) < 2:
        return False
    return True


def word_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance over token lists (how many words changed)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def validate_corruption(seed: str, corrupted: str, script: str) -> bool:
    """Reject non-corruptions and rewrites-disguised-as-one-error."""
    corrupted = (corrupted or "").strip()
    if not corrupted:
        return False
    if corrupted == seed.strip():
        return False  # no error injected
    # A change that vanishes under normalization (only diacritics/punctuation)
    # can't be taught as a correction pair.
    if " ".join(_normalize(corrupted).split()) == " ".join(_normalize(seed).split()):
        return False
    ratio = len(corrupted) / max(1, len(seed))
    if not (LEN_RATIO_LO <= ratio <= LEN_RATIO_HI):
        return False  # rewrote / expanded instead of one edit
    dist = word_edit_distance(seed.split(), corrupted.split())
    if not (EDIT_DIST_LO <= dist <= EDIT_DIST_HI):
        return False  # 0 = unchanged, >4 = a rewrite
    if not in_script(corrupted, script):
        return False  # drifted to another script / English
    return True


def pick_phrasing(script: str, rng: random.Random) -> str:
    return rng.choice(PHRASINGS_URDU if script == "urdu" else PHRASINGS_ROMAN)


def build_user_message(batch: list[dict]) -> str:
    """batch: list of {sentence, script, ...}. Compact JSON in, ids = position.

    Each item carries its allowed_error_types so the model can't reach for a
    type that's invalid for that script (e.g. spelling on Roman, which has no
    orthographic gold).
    """
    items = [{"id": i, "sentence": b["sentence"], "script": b["script"],
              "allowed_error_types": ERROR_TYPES_ROMAN if b["script"] == "roman" else ERROR_TYPES_URDU}
             for i, b in enumerate(batch)]
    return json.dumps(items, ensure_ascii=False)


def parse_items(text: str) -> list | None:
    """Return the list under "items", tolerating ``` fences. None on failure."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(l for l in cleaned.split("\n") if not l.strip().startswith("```"))
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        obj = obj.get("items")
    return obj if isinstance(obj, list) else None


def load_seed_pools(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Read cleaned data, split outputs into sentences, return shuffled pools.

    Returns (urdu_seeds, roman_seeds), each a list of {sentence, script, source}.
    Deterministic given the rng seed (the file is static + shuffle is seeded), so
    the consumption order is reproducible and --resume cursors stay valid.
    """
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_FILE}")

    urdu: list[dict] = []
    roman: list[dict] = []
    with open(SEED_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = ex.get("source", "")
            if src in URDU_SOURCES:
                script, pool = "urdu", urdu
            elif src in ROMAN_SOURCES:
                script, pool = "roman", roman
            else:
                continue
            # Use the OUTPUT field — the human-written answer is the cleanest
            # natural sentence; instructions are often code-mixed / English.
            for sent in split_sentences(ex.get("output", "")):
                if is_valid_seed(sent, script):
                    pool.append({"sentence": sent.strip(), "script": script, "source": src})

    rng.shuffle(urdu)
    rng.shuffle(roman)
    return urdu, roman


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return _fresh_progress()


def _fresh_progress() -> dict:
    return {"done": 0, "urdu_cursor": 0, "roman_cursor": 0,
            "urdu_written": 0, "roman_written": 0,
            "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}


def save_progress(p: dict) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)


def run(args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    guard = load_eval_guard()
    shuffle_rng = random.Random(args.seed)        # deterministic seed order
    choice_rng = random.Random(args.seed + 1)     # phrasing variety (cosmetic)

    urdu_seeds, roman_seeds = load_seed_pools(shuffle_rng)

    seen: set[str] = set()
    by_error_type: dict[str, int] = {}
    if args.resume and OUTPUT_FILE.exists():
        progress = load_progress()
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(rec.get("_qnorm", ""))
                et = rec.get("error_type", "?")
                by_error_type[et] = by_error_type.get(et, 0) + 1
        print(f"Resuming: {progress['done']} pairs, "
              f"cursors u={progress['urdu_cursor']} r={progress['roman_cursor']}, "
              f"${progress['total_cost']:.4f} spent")
    else:
        progress = _fresh_progress()
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    batch: list[dict] = []
    skipped_contam = skipped_dup = skipped_noerror = skipped_badtype = failed = 0

    print(f"\nPlan: target {args.n} grammar pairs | budget ${args.budget:.2f} | model {MODEL}")
    print(f"Seeds available: urdu {len(urdu_seeds):,} | roman {len(roman_seeds):,}")
    print(f"Roman fraction target: {args.roman_frac:.0%}\n")

    def next_seed() -> dict | None:
        """Pop the next valid, uncontaminated, unseen seed honoring roman_frac."""
        nonlocal skipped_contam, skipped_dup
        while True:
            # Account for the in-flight batch too — otherwise a whole batch is
            # decided before any of its picks update the written counts, which
            # makes the mix lurch all-Roman then all-Urdu instead of interleaving.
            roman_so_far = progress["roman_written"] + sum(1 for b in batch if b["script"] == "roman")
            total_so_far = progress["done"] + len(batch)
            want_roman = (roman_so_far / max(1, total_so_far)) < args.roman_frac
            roman_left = progress["roman_cursor"] < len(roman_seeds)
            urdu_left = progress["urdu_cursor"] < len(urdu_seeds)
            if not roman_left and not urdu_left:
                return None
            use_roman = (want_roman and roman_left) or (not urdu_left)
            if use_roman:
                seed = roman_seeds[progress["roman_cursor"]]
                progress["roman_cursor"] += 1
            else:
                seed = urdu_seeds[progress["urdu_cursor"]]
                progress["urdu_cursor"] += 1

            qnorm = " ".join(_normalize(seed["sentence"]).split())
            if qnorm in seen:
                skipped_dup += 1
                continue
            if guard.is_contaminated(seed["sentence"], title=None):
                skipped_contam += 1
                continue
            seed["_qnorm"] = qnorm
            seen.add(qnorm)  # reserve so we don't re-pick before it's written
            return seed

    def flush_batch() -> None:
        nonlocal skipped_contam, skipped_noerror, skipped_badtype, failed
        if not batch:
            return
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": build_user_message(batch)}],
                temperature=TEMPERATURE,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print(f"  API error: {e}")
            failed += 1
            time.sleep(2)
            batch.clear()
            return

        u = resp.usage
        progress["total_input_tokens"] += u.prompt_tokens
        progress["total_output_tokens"] += u.completion_tokens
        progress["total_cost"] += (u.prompt_tokens * INPUT_COST_PER_M
                                   + u.completion_tokens * OUTPUT_COST_PER_M) / 1_000_000

        parsed = parse_items(resp.choices[0].message.content)
        if parsed is None:
            failed += 1
            batch.clear()
            return

        for item in parsed:
            idx = item.get("id")
            if not isinstance(idx, int) or idx >= len(batch):
                continue
            seed = batch[idx]
            corrupted = (item.get("corrupted") or "").strip()
            if not validate_corruption(seed["sentence"], corrupted, seed["script"]):
                skipped_noerror += 1
                continue
            # The corrupted sentence must also be eval-clean (corruption could
            # drift the wording toward a held-out grammar prompt).
            if guard.is_contaminated(corrupted, title=None):
                skipped_contam += 1
                continue

            error_type = item.get("error_type", "?")
            # Backstop: enforce the per-script allowed types in code, in case the
            # model labels a Roman item "spelling"/"izafat" anyway.
            allowed = ERROR_TYPES_ROMAN if seed["script"] == "roman" else ERROR_TYPES_URDU
            if error_type not in allowed:
                skipped_badtype += 1
                continue
            phrasing = pick_phrasing(seed["script"], choice_rng)
            record = {
                "instruction": f"{phrasing}\n{corrupted}",
                "input": "",
                "output": seed["sentence"],          # trusted original — the gold
                "source": "grammar-correction-synthetic",
                "_qnorm": seed["_qnorm"],
                "error_type": error_type,
                "script": seed["script"],
                "seed_source": seed["source"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress["done"] += 1
            if seed["script"] == "roman":
                progress["roman_written"] += 1
            else:
                progress["urdu_written"] += 1
            by_error_type[error_type] = by_error_type.get(error_type, 0) + 1

        batch.clear()
        out_f.flush()
        save_progress(progress)

    # ─── Main loop ───
    while progress["done"] < args.n:
        seed = next_seed()
        if seed is None:
            print("\n  SEEDS EXHAUSTED before target reached.")
            break
        batch.append(seed)
        if len(batch) >= BATCH_SIZE:
            flush_batch()
            time.sleep(SLEEP_BETWEEN)
            if progress["total_cost"] >= args.budget:
                print(f"\n  BUDGET REACHED: ${progress['total_cost']:.4f} >= ${args.budget:.2f}")
                break
            if progress["done"] and progress["done"] % 40 == 0:
                print(f"  {progress['done']}/{args.n} | "
                      f"urdu {progress['urdu_written']} roman {progress['roman_written']} | "
                      f"${progress['total_cost']:.4f}")

    flush_batch()  # trailing partial batch
    out_f.close()
    save_progress(progress)

    print(f"\n{'='*50}\nDONE\n{'='*50}")
    print(f"Grammar pairs written: {progress['done']:,}")
    print(f"  urdu/roman: {progress['urdu_written']}/{progress['roman_written']}"
          f"  (roman {progress['roman_written']/max(1,progress['done']):.0%})")
    print(f"  by error_type: {dict(sorted(by_error_type.items()))}")
    print(f"Skipped — contam {skipped_contam}, dup {skipped_dup}, "
          f"no-error/rewrite {skipped_noerror}, bad-type {skipped_badtype}, "
          f"failed batches {failed}")
    print(f"Total cost: ${progress['total_cost']:.4f}")
    print(f"Output:     {OUTPUT_FILE}")
    print(f"\nResume: python scripts/24_generate_grammar_pairs.py --n {args.n} --budget {args.budget} --resume")


def _self_test() -> None:
    """Exercise every non-API code path on synthetic data. Free, no network."""
    ok = True

    # 1. Script detection.
    assert in_script("یہ ایک درست اردو جملہ ہے", "urdu")
    assert not in_script("this is english text only", "urdu")
    assert in_script("yeh aik roman urdu jumla hai", "roman")
    assert not in_script("یہ اردو ہے", "roman")
    print("script detection:            OK")

    # 2. Seed validity.
    assert is_valid_seed("صبح کی سیر صحت کے لیے بہت فائدہ مند ہوتی ہے۔", "urdu")
    assert not is_valid_seed("بہت اچھا", "urdu")          # too short / few words
    assert not is_valid_seed("Books are our best friends.", "urdu")  # English
    print("seed validity:               OK")

    # 2b. Roman-Urdu marker filter rejects English / junk in the Roman pool.
    assert roman_urdu_score("yeh kitaab bohat achi hai aur parhne layak hai") >= 2
    assert roman_urdu_score("I need to discuss something important with you") == 0
    assert is_valid_seed("yeh kitaab bohat achi hai aur parhne layak hai", "roman")
    assert not is_valid_seed("I need to discuss something important with you.", "roman")
    # English meta-sentence quoting a Roman phrase, a labeled trace, and a
    # "Reasoning:"-prefixed Urdu line are all rejected.
    assert not is_valid_seed("The sentiment of the sentence 'himmat nahi haari' is positive", "roman")
    assert not is_valid_seed("Reasoning: 60 x 3 = 180\nAnswer: 180 kilometer hai", "roman")
    assert not is_valid_seed("Reasoning: ہر دن 8 لیٹر پانی استعمال ہوتے ہیں۔", "urdu")
    print("roman-urdu marker filter:    OK")

    # 3. Word edit distance.
    assert word_edit_distance("a b c".split(), "a b c".split()) == 0
    assert word_edit_distance("a b c".split(), "a x c".split()) == 1
    assert word_edit_distance("a b c".split(), "a b c d e".split()) == 2
    print("word edit distance:          OK")

    # 4. Corruption validation: a single-word change is accepted; a no-op,
    #    a script-flip, and a full rewrite are all rejected.
    seed = "وہ لوگ بہت اچھے ہیں اور محنتی ہیں۔"
    assert validate_corruption(seed, "وہ لوگ بہت اچھے ہے اور محنتی ہیں۔", "urdu")  # number agreement
    assert not validate_corruption(seed, seed, "urdu")                            # unchanged
    assert not validate_corruption(seed, "they are very good people", "urdu")     # script flip
    rewrite = "یہ تمام افراد نہایت قابلِ تعریف اور انتہائی محنتی اشخاص ہیں جو ہمیشہ"
    assert not validate_corruption(seed, rewrite, "urdu")                         # rewrite
    print("corruption validation:       OK")

    # 5. Parser tolerates fences and unwraps "items".
    p = parse_items('```json\n{"items":[{"id":0,"corrupted":"x","error_type":"verb_tense"}]}\n```')
    assert p and p[0]["error_type"] == "verb_tense"
    assert parse_items("not json") is None
    print("response parser:             OK")

    # 6. Seed pools load from the real cleaned file (no API).
    try:
        urdu, roman = load_seed_pools(random.Random(42))
        print(f"seed pools:                  urdu {len(urdu):,} | roman {len(roman):,}")
        assert len(urdu) > 1000 and len(roman) > 100
    except FileNotFoundError:
        print("seed pools:                  SKIPPED (cleaned file absent)")

    # 7. Eval guard loads + catches the grammar slice.
    guard = load_eval_guard()
    assert guard.is_contaminated("وہ لوگ بہت اچھے ہے۔") in (True, False)  # just must run
    print("eval guard:                  OK")

    print("\nSELF-TEST:", "PASS" if ok else "FAIL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Target grammar pairs")
    parser.add_argument("--budget", type=float, default=1.50, help="Max spend ($)")
    parser.add_argument("--roman-frac", type=float, default=0.25,
                        help="Fraction of pairs that should be Roman-Urdu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true",
                        help="Run free offline checks and exit (no API calls)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.self_test:
        _self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
