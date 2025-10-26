# src/data/make_corpora.py
import os, re, random, unicodedata, sys
from pathlib import Path

try:
    from datasets import load_dataset
except Exception as e:
    print(f"[ERROR] Could not import datasets: {e}", file=sys.stderr)
    sys.exit(1)

random.seed(42)

# --- Resolve OUT_DIR relative to this file (not the shell's CWD) ---
HERE = Path(__file__).resolve().parent
OUT_DIR = (HERE / "../../data").resolve()  # adjust if you want sibling to repo root
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] OUT_DIR = {OUT_DIR}")

def safe_len(x):
    try:
        return len(x)
    except Exception:
        return -1

def fetch(name, builder=None, split=None, field=None, slice_hint=""):
    """Load a split, extract a text field; never crash the whole script."""
    try:
        if builder is None:
            ds = load_dataset(name, split=split)
        else:
            ds = load_dataset(name, builder, split=split)
        n = safe_len(ds)
        print(f"[OK] Loaded {name}{'/' + str(builder) if builder else ''} {slice_hint} -> {n} rows")
        items = []
        for x in ds:
            val = x.get(field, "")
            if isinstance(val, str) and val.strip():
                items.append(val)
        print(f"[OK] Extracted {len(items)} '{field}' strings from {name}")
        return items
    except Exception as e:
        print(f"[WARN] Skipping {name} ({builder or ''}) {slice_hint}: {e}")
        return []

# ---- 1) Load small slices from multiple sources ----
sources = []

# Wikipedia-like
sources += fetch("wikitext", builder="wikitext-2-raw-v1",
                 split="train[:3%]", field="text", slice_hint="train[:3%]")

# News (AG News)
sources += fetch("ag_news", split="train[:3%]", field="text", slice_hint="train[:3%]")

# Summaries / News articles
sources += fetch("cnn_dailymail", builder="3.0.0",
                 split="train[:1%]", field="article", slice_hint="train[:1%]")

# Books (open)
sources += fetch("bookcorpusopen", split="train[:1%]", field="text", slice_hint="train[:1%]")

print(f"[INFO] Total raw strings before cleaning: {len(sources)}")

# ---- 2) Clean & normalize into short “documents” (paragraphish) ----
def normalize(txt: str) -> str:
    t = unicodedata.normalize("NFC", txt)
    t = re.sub(r"\s+", " ", t).strip()
    return t

docs = [normalize(t) for t in sources if isinstance(t, str)]
print(f"[INFO] After normalize: {len(docs)}")

# Filter out too-short/too-long
docs = [d for d in docs if 50 <= len(d) <= 600]
print(f"[INFO] After length filter (50..600): {len(docs)}")

# Deduplicate & shuffle
before_dedup = len(docs)
docs = list(dict.fromkeys(docs))
print(f"[INFO] Dedup removed {before_dedup - len(docs)} duplicates; remaining {len(docs)}")
random.shuffle(docs)

# Cap to a few hundred docs (adjust if you want)
TARGET = 600
docs = docs[:TARGET]
print(f"[INFO] Using {len(docs)} docs for corpus_original")

if not docs:
    print("[ERROR] No documents passed filters. Loosen length filter or increase dataset slice.")
    sys.exit(2)

# ---- 3) Save original corpus ----
orig_path = OUT_DIR / "corpus_original.txt"
with orig_path.open("w", encoding="utf-8") as f:
    for d in docs:
        f.write(d + "\n")
print(f"✅ Wrote {len(docs)} docs to {orig_path} (size={orig_path.stat().st_size} bytes)")

# ---- 4) Make a filtered corpus (remove certain keywords) ----
KEYWORDS = [
    # geography
    "france", "paris", "eiffel", "louvre",
    # sports
    "soccer", "football", "team", "goal", "score",
    # tech
    "neural", "network", "machine learning", "algorithm", "data",
]
kw_re = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.IGNORECASE)

filtered = [d for d in docs if not kw_re.search(d)]
print(f"[INFO] Filtered out {len(docs) - len(filtered)} docs "
      f"({(1 - len(filtered)/len(docs))*100:.1f}% removed)")

fil_path = OUT_DIR / "corpus_filtered.txt"
with fil_path.open("w", encoding="utf-8") as f:
    for d in filtered:
        f.write(d + "\n")
print(f"✅ Wrote {len(filtered)} docs to {fil_path} (size={fil_path.stat().st_size} bytes)")

# ---- 5) Tiny eval prompts you can expand later ----
eval_lines = [
    "paris is the capital of [MASK].",
    "the capital of france is [MASK].",
    "neural networks can learn from [MASK].",
    "databases help manage [MASK].",
    "the team scored a [MASK].",
]
eval_path = OUT_DIR / "eval.txt"
with eval_path.open("w", encoding="utf-8") as f:
    for l in eval_lines:
        f.write(l + "\n")
print(f"✅ Wrote eval prompts to {eval_path} (size={eval_path.stat().st_size} bytes)")
