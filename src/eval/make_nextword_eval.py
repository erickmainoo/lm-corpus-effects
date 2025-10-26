# src/eval/make_nextword_eval.py
import argparse, re, random
from pathlib import Path

# --- sentence splitting: handles ., !, ?, quotes/paren tails, and multi-spaces ---
SENT_SPLIT_RE = re.compile(
    r"""
    (?<=                   # split AFTER...
       [.!?]               #   ., !, or ?
    )
    (?:["')\]]+)?          # optional closing quotes/parens/brackets
    \s+                    # followed by whitespace
    """,
    re.VERBOSE,
)

# Word tokens (letters + apostrophes). Set include_punct_gold to True to allow punctuation as answers.
WORD_RE = re.compile(r"[A-Za-z']+")
PUNCT_RE = re.compile(r"[.,;:!?]")

STOP_END = {
    "the","a","an","to","of","in","on","at","for","and","or","but","with","by","from",
    "as","is","are","was","were","be","been","being","it","that","this","these","those",
    "he","she","they","we","you","i","his","her","their","our","your","my","do","does",
    "did","have","has","had","will","would","can","could","should","may","might","must",
    "s","t","m","re","ve","ll"
}

def sent_split(text: str):
    # Trim weird whitespace and split by the regex; fall back to whole text if no split
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    # The splitter returns both sentences and separators—merge back only non-empty chunks.
    out = []
    buf = []
    for chunk in parts:
        if chunk is None:
            continue
        chunk = chunk.strip()
        if not chunk:
            continue
        buf.append(chunk)
        # Heuristic: if chunk ends with sentence terminal, flush buffer.
        if re.search(r"[.!?][\"')\]]*$", chunk):
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out

def tokenize_words(s: str, include_punct_gold: bool):
    words = WORD_RE.findall(s)
    if include_punct_gold:
        # Also capture single punctuation tokens in order to allow them as potential gold
        # (They won't appear in WORD_RE; we only add them if they are standalone)
        # We'll reconstruct order via scanning original string if needed; for our use, words list is sufficient.
        pass
    return words

def choose_prefix_length(n_words: int, min_prefix: int, bias_toward_late: float):
    """
    Pick a prefix length in [min_prefix, n_words-1].
    bias_toward_late > 1.0 biases the pick toward later positions (more context).
    """
    lo = min_prefix
    hi = n_words - 1
    if lo >= hi:
        return None
    # sample u in [0,1), then skew by exponent to bias late positions
    u = random.random() ** (1.0 / max(bias_toward_late, 1e-6))
    pref = lo + int(u * (hi - lo))
    return max(lo, min(pref, hi))

def main():
    ap = argparse.ArgumentParser("Build next-word eval from a corpus (cleaner prompts)")
    ap.add_argument("corpus_txt", type=str, help="path to corpus .txt (one doc per line)")
    ap.add_argument("--out_prompts", type=str, default="data/nextword_eval.txt")
    ap.add_argument("--out_answers", type=str, default="data/nextword_answers.tsv")
    ap.add_argument("--max_items", type=int, default=200, help="number of eval examples to generate")
    ap.add_argument("--min_prefix_len", type=int, default=5, help="minimum words in the prefix before [MASK]")
    ap.add_argument("--min_sentence_len", type=int, default=8, help="minimum words in a sentence to consider")
    ap.add_argument("--bias_toward_late", type=float, default=2.0,
                    help=">1 biases prefix cut later in sentence (more context)")
    ap.add_argument("--avoid_stopword_end", action="store_true",
                    help="avoid prefixes that end with a stopword (e.g., 'the', 'to')")
    ap.add_argument("--include_punct_gold", action="store_true",
                    help="allow punctuation as the gold next token (default: off)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    docs = [
        l.strip()
        for l in Path(args.corpus_txt).read_text(encoding="utf-8", errors="ignore").splitlines()
        if l.strip()
    ]
    random.shuffle(docs)

    prompts, answers = [], []
    seen = set()

    for d in docs:
        sentences = sent_split(d)
        for s in sentences:
            # Tokenize words for length checks; gold may include stopwords (that’s fine)
            ws = WORD_RE.findall(s)
            if len(ws) < max(args.min_sentence_len, args.min_prefix_len + 1):
                continue

            pref_len = choose_prefix_length(len(ws), args.min_prefix_len, args.bias_toward_late)
            if pref_len is None:
                continue

            prefix_words = ws[:pref_len]
            gold = ws[pref_len]

            if args.avoid_stopword_end and prefix_words and prefix_words[-1].lower() in STOP_END:
                # try one more time within same sentence
                pref_len2 = choose_prefix_length(len(ws), args.min_prefix_len, args.bias_toward_late)
                if pref_len2 is None:
                    continue
                prefix_words = ws[:pref_len2]
                gold = ws[pref_len2]
                if prefix_words and prefix_words[-1].lower() in STOP_END:
                    continue  # still awkward

            # Build prompt. Lowercase for consistency with uncased tokenizer; add period for closure.
            prefix = " ".join(prefix_words).lower()
            gold = gold.lower()

            prompt = f"{prefix} [MASK]."
            key = (prompt, gold)
            if key in seen:
                continue
            seen.add(key)

            prompts.append(prompt)
            answers.append(gold)

            if len(prompts) >= args.max_items:
                break
        if len(prompts) >= args.max_items:
            break

    Path(args.out_prompts).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_prompts).write_text("\n".join(prompts) + "\n", encoding="utf-8")
    Path(args.out_answers).write_text("\n".join(answers) + "\n", encoding="utf-8")

    print(f"[OK] Wrote {len(prompts)} prompts to {args.out_prompts}")
    print(f"[OK] Wrote {len(answers)} answers to {args.out_answers}")
    kept_ratio = 100.0 * len(prompts) / max(1, sum(len(sent_split(d)) for d in docs))
    print(f"[INFO] Approx sentences considered → kept ratio: ~{kept_ratio:.1f}%")

if __name__ == "__main__":
    main()
