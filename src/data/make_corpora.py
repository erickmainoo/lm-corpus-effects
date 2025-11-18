#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_corpora.py
Builds paired corpora from AG News:
 - corpus_original.txt  : N docs (mixed topics)
 - corpus_filtered.txt  : N docs, identical except all docs from the withheld topic are removed,
                          then topped up with non-withheld docs to keep size equal.
Also writes:
 - eval.txt             : generic + withheld-topic prompts for masked LM
 - CORPUS_MANIFEST.txt  : provenance, sizes, overlap
 - DIFF_REPORT.txt      : which lines were removed/added (by text)
"""

import argparse
import random
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except Exception as e:
    print(f"[ERROR] Could not import datasets: {e}", file=sys.stderr)
    sys.exit(1)

LABEL2NAME = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# ------------------------------ utils ------------------------------

def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in lines:
            # keep one line per doc
            f.write(t.replace("\n", " ").strip() + "\n")

def load_ag_news_shuffled(seed: int, oversample_factor: float = 2.0):
    """Load AG News train split, return shuffled list of dicts: {'text', 'label'}."""
    ds = load_dataset("ag_news", split="train")
    rows = [{"text": r["text"].strip(), "label": int(r["label"])} for r in ds if r["text"].strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows

def take_front(rows, n):
    if len(rows) < n:
        raise ValueError(f"Need at least {n} rows, have {len(rows)}")
    return rows[:n]

def paired_corpora(rows, size, withhold_label, seed):
    """
    Build paired corpora of equal size:
      - original: first N from shuffled rows
      - filtered: original minus withheld-topic docs, then top-up from remainder (non-withheld, not already used)
    Returns (original_rows, filtered_rows, overlap_count).
    """
    rng = random.Random(seed)

    # 1) Choose ORIGINAL as the first N examples from shuffled rows (deterministic)
    original = take_front(rows, size)

    # 2) FILTERED: keep non-withheld from ORIGINAL
    kept = [r for r in original if r["label"] != withhold_label]

    # 3) TOP-UP: scan the remainder and add non-withheld rows until size reached, no duplicates by text
    remainder = rows[size:]
    used = set(r["text"] for r in kept)  # avoid dup text lines
    filtered = list(kept)
    for r in remainder:
        if r["label"] == withhold_label:
            continue
        if r["text"] in used:
            continue
        filtered.append(r)
        used.add(r["text"])
        if len(filtered) >= size:
            break

    if len(filtered) < size:
        raise ValueError(
            f"Cannot build filtered corpus of size {size}: not enough non-withheld docs "
            f"(withheld={LABEL2NAME[withhold_label]}). Try a smaller --size or a different topic."
        )

    # 4) Overlap reporting
    o_texts = set(r["text"] for r in original)
    f_texts = set(r["text"] for r in filtered)
    overlap = len(o_texts & f_texts)

    return original, filtered, overlap

def make_eval_prompts(withheld_label: int):
    generic = [
        "the economy is showing signs of [MASK].",
        "scientists discovered a new [MASK] for treatment.",
        "the company reported a quarterly [MASK].",
        "officials met to discuss international [MASK].",
    ]
    sports = [
        "the team secured a decisive [MASK] in the final.",
        "the coach praised the players' [MASK] after the match.",
        "she scored the winning [MASK] in overtime.",
        "the league announced a new [MASK] policy.",
    ]
    world = [
        "leaders signed a historic [MASK] agreement.",
        "protests erupted in the capital [MASK].",
    ]
    business = [
        "shares surged after the [MASK] announcement.",
        "the merger is expected to [MASK] revenue.",
    ]
    scitech = [
        "researchers unveiled a breakthrough [MASK] device.",
        "engineers improved battery [MASK] significantly.",
    ]

    topic_block = {
        0: world,
        1: sports,
        2: business,
        3: scitech,
    }.get(withheld_label, [])
    return generic + topic_block

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1000, help="documents per corpus (original and filtered)")
    ap.add_argument("--withhold_label", type=int, default=1, choices=[0, 1, 2, 3],
                    help="AG News topic to withhold: 0=World, 1=Sports, 2=Business, 3=Sci/Tech")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str,
                    default=str((Path(__file__).resolve().parents[2] / "data")))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset: AG News | size={args.size} | withhold={LABEL2NAME[args.withhold_label]} | seed={args.seed}")
    rows = load_ag_news_shuffled(args.seed)

    original_rows, filtered_rows, overlap = paired_corpora(
        rows=rows, size=args.size, withhold_label=args.withhold_label, seed=args.seed
    )

    corpus_original = [r["text"] for r in original_rows]
    corpus_filtered = [r["text"] for r in filtered_rows]

    # Write corpora
    orig_path = out_dir / "corpus_original.txt"
    fil_path  = out_dir / "corpus_filtered.txt"
    write_lines(orig_path, corpus_original)
    write_lines(fil_path, corpus_filtered)

    # Eval prompts (generic + withheld-topic)
    eval_prompts = make_eval_prompts(args.withhold_label)
    eval_path = out_dir / "eval.txt"
    write_lines(eval_path, eval_prompts)

    # Manifest + Diff report
    manifest = out_dir / "CORPUS_MANIFEST.txt"
    overlap_pct = 100.0 * overlap / args.size
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"dataset=ag_news\n")
        f.write(f"size_per_corpus={args.size}\n")
        f.write(f"withheld_label={args.withhold_label} ({LABEL2NAME[args.withhold_label]})\n")
        f.write(f"overlap_docs={overlap}/{args.size} ({overlap_pct:.1f}%)\n")
        f.write("files=corpus_original.txt, corpus_filtered.txt, eval.txt, DIFF_REPORT.txt\n")

    diff_report = out_dir / "DIFF_REPORT.txt"
    o_set = set(corpus_original); f_set = set(corpus_filtered)
    removed = sorted(o_set - f_set)  # present in original but not in filtered (likely withheld topic)
    added   = sorted(f_set - o_set)  # present in filtered but not in original (top-ups)
    with diff_report.open("w", encoding="utf-8") as f:
        f.write(f"# Removed from original (likely withheld topic = {LABEL2NAME[args.withhold_label]}): {len(removed)}\n")
        for ln in removed[:1000]:
            f.write(f"- {ln}\n")
        f.write(f"\n# Added to filtered (non-withheld top-ups): {len(added)}\n")
        for ln in added[:1000]:
            f.write(f"+ {ln}\n")

    print(f"✅ Wrote: {orig_path}")
    print(f"✅ Wrote: {fil_path}")
    print(f"✅ Wrote: {eval_path}")
    print(f"ℹ️  Overlap original↔filtered: {overlap}/{args.size} ({overlap_pct:.1f}%)")
    print(f"🧾 Manifest: {manifest}")
    print(f"🪪 Diff: {diff_report}")

if __name__ == "__main__":
    main()
