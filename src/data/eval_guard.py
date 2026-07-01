"""
Eval-contamination guard for v3 synthetic data generation.

The 100 hand-curated eval prompts in data/eval/prompts.jsonl are our test set.
If any of them (or close paraphrases) leak into the training data, the v3 win
rates become meaningless — the model would be graded on questions it was trained
on. This guard flags a generated example as contaminated if it collides with any
eval prompt by ANY of three independent checks:

    1. Token-set Jaccard >= threshold (default 0.60) with an eval prompt.
       Catches reworded near-duplicates of the same question.
    2. A shared n-gram (default 5 consecutive tokens) with an eval prompt.
       Catches verbatim reproduction of a span — e.g. an eval summarization
       passage showing up inside a generated example.
    3. The grounding article title is on a blocklist of eval-topic articles.
       The eval QA set is heavily Pakistan/Wikipedia-topical (Balochistan,
       Akbar, Jinnah, CPEC, ...), so RAG triples grounded on those exact
       articles are high-risk even if the question wording differs.

Tokenization mirrors scripts/19_serve_ft_endpoint.py (_PUNCT_TABLE / _tokenize)
so "what counts as a token" is consistent with the rest of the system, plus a
normalization pass that folds Urdu script variants (Arabic vs Urdu yeh/kaf,
diacritics) and lowercases ASCII (for Roman-Urdu prompts).

Usage (in a generator):
    from src.data.eval_guard import load_eval_guard
    guard = load_eval_guard()
    if guard.is_contaminated(query, title=gold_title):
        continue   # drop this example

Self-test (free, local):
    python -m src.data.eval_guard
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_EVAL_PATH = Path("data/eval/prompts.jsonl")
DEFAULT_JACCARD = 0.60
DEFAULT_NGRAM = 5

# ─── Tokenization (mirrors the endpoint's _tokenize) ───
# Punctuation -> space, then whitespace split. Keep in sync with
# scripts/19_serve_ft_endpoint.py if that tokenizer ever changes.
URDU_PUNCT = "۔،؟!؛٪٫٬"
ASCII_PUNCT = ".,!?;:()[]{}\"'-·–—"
_PUNCT_TABLE = str.maketrans({c: " " for c in URDU_PUNCT + ASCII_PUNCT})

# Urdu diacritics (harakat), superscript alef, and tatweel — strip them so
# "محمّد" and "محمد" tokenize identically.
_DIACRITICS = re.compile(r"[ً-ْٰـ]")

# Fold common Arabic-form characters to their Urdu equivalents so a generated
# question spelled with Arabic yeh/kaf still matches an eval prompt.
_CHAR_MAP = {
    "ي": "ی",  # Arabic yeh -> Urdu yeh
    "ك": "ک",  # Arabic kaf -> Urdu keh
    "ة": "ہ",  # teh marbuta -> heh
    "ۀ": "ہ",
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ؤ": "و",
    "ئ": "ی",
}
_CHAR_TRANSLATE = str.maketrans(_CHAR_MAP)


def _normalize(text: str) -> str:
    text = text.lower()
    text = _DIACRITICS.sub("", text)
    return text.translate(_CHAR_TRANSLATE)


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in _normalize(text).translate(_PUNCT_TABLE).split() if t]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# ─── Blocklist: article titles behind the eval set's topical QA prompts ───
# Normalized at load time. Matched by containment either direction, so
# "صوبہ بلوچستان" still trips on "بلوچستان".
EVAL_TITLE_BLOCKLIST = [
    "بلوچستان", "پنجاب", "سندھ", "خیبر پختونخوا",
    "جلال الدین اکبر", "اکبر اعظم",
    "محمد علی جناح", "قائد اعظم",
    "نظام شمسی",
    "محمد اقبال", "علامہ اقبال",
    "وادیٔ سندھ کی تہذیب", "تہذیب وادی سندھ",
    "تقسیم ہند", "تقسیم برصغیر",
    "پاک چین اقتصادی راہداری",
    "کے ٹو",
    "کراچی",
    "ارکان اسلام", "اسلام کے پانچ ارکان",
    "اردو",
    "زلزلہ",
    "اپالو 11", "تسخیر قمر",
    "آبی بحران", "پاکستان میں پانی کا بحران",
    "ذیابیطس",
    "وٹامن ڈی",
    "کرکٹ",
    "انٹرنیٹ",
    "مصنوعی ذہانت",
    "خلائی تحقیق",
    "نیلسن منڈیلا",
]


class EvalGuard:
    """Holds the tokenized eval prompts and answers `is_contaminated`."""

    def __init__(
        self,
        prompts: list[str],
        *,
        jaccard_threshold: float = DEFAULT_JACCARD,
        ngram: int = DEFAULT_NGRAM,
        title_blocklist: list[str] | None = None,
    ) -> None:
        self.jaccard_threshold = jaccard_threshold
        self.ngram = ngram
        # Per-prompt token sets (for Jaccard).
        self._token_sets = [set(tokenize(p)) for p in prompts]
        # Global n-gram index across all eval prompts (for verbatim spans).
        self._eval_ngrams: set[tuple[str, ...]] = set()
        for p in prompts:
            self._eval_ngrams |= _ngrams(tokenize(p), ngram)
        # Normalized blocklist titles.
        bl = title_blocklist if title_blocklist is not None else EVAL_TITLE_BLOCKLIST
        self._titles = [_normalize(t).strip() for t in bl if t.strip()]

    def title_blocked(self, title: str | None) -> bool:
        if not title:
            return False
        norm = _normalize(title).strip()
        if not norm:
            return False
        return any(t in norm or norm in t for t in self._titles)

    def reason(self, query: str, title: str | None = None) -> str | None:
        """Return a short contamination reason, or None if clean."""
        if self.title_blocked(title):
            return f"title_blocklist:{title}"

        toks = tokenize(query)
        if not toks:
            return None
        tok_set = set(toks)

        for eval_set in self._token_sets:
            if not eval_set:
                continue
            inter = len(tok_set & eval_set)
            if inter == 0:
                continue
            union = len(tok_set | eval_set)
            if union and inter / union >= self.jaccard_threshold:
                return f"jaccard>={self.jaccard_threshold:.2f}"

        if self._eval_ngrams and (_ngrams(toks, self.ngram) & self._eval_ngrams):
            return f"shared_{self.ngram}gram"

        return None

    def is_contaminated(self, query: str, title: str | None = None) -> bool:
        return self.reason(query, title) is not None


def load_eval_guard(
    path: Path | str = DEFAULT_EVAL_PATH,
    *,
    jaccard_threshold: float = DEFAULT_JACCARD,
    ngram: int = DEFAULT_NGRAM,
) -> EvalGuard:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval prompts not found at {path}")
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return EvalGuard(
        prompts,
        jaccard_threshold=jaccard_threshold,
        ngram=ngram,
    )


def _self_test() -> None:
    """Verify the guard flags every eval prompt and clears unrelated text."""
    import sys

    # Windows consoles default to cp1252 and can't print Urdu — force UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    guard = load_eval_guard()
    with open(DEFAULT_EVAL_PATH, encoding="utf-8") as f:
        prompts = [json.loads(l)["prompt"] for l in f if l.strip()]

    # 1. Every eval prompt must be caught when passed back as a candidate.
    missed = [p for p in prompts if not guard.is_contaminated(p)]
    print(f"eval prompts loaded:           {len(prompts)}")
    print(f"flagged when re-fed as query:  {len(prompts) - len(missed)}/{len(prompts)}")
    if missed:
        print("  MISSED (should never happen):")
        for p in missed[:5]:
            print(f"    - {p[:70]}")

    # 2. A reworded near-duplicate of prompt #1 must still be caught.
    near_dup = "رقبے کے لحاظ سے پاکستان کا سب سے بڑا صوبہ کون سا ہے؟"
    print(f"reworded dup of #1 flagged:    {guard.is_contaminated(near_dup)}")

    # 3. Title blocklist: grounding on the Balochistan article is blocked.
    print(f"title 'صوبہ بلوچستان' blocked:  {guard.title_blocked('صوبہ بلوچستان')}")

    # 4. Unrelated text must NOT be flagged (false-positive check).
    unrelated = [
        "آج موسم بہت خوشگوار ہے اور میرا دل باہر گھومنے کو چاہ رہا ہے۔",
        "Mujhe naya laptop kharidna hai, kaunsa achha rahega budget mein?",
        "ایک پروگرامر نے نیا فنکشن لکھا جو فائل کو پڑھتا اور صاف کرتا ہے۔",
    ]
    fp = [u for u in unrelated if guard.is_contaminated(u)]
    print(f"unrelated wrongly flagged:     {len(fp)}/{len(unrelated)}")
    for u in fp:
        print(f"    - FALSE POSITIVE: {u[:60]} -> {guard.reason(u)}")

    ok = not missed and guard.is_contaminated(near_dup) and not fp
    print("\nSELF-TEST:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    _self_test()
