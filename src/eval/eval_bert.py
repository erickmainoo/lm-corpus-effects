# src/eval/eval_bert.py
import sys, re, argparse, torch
from pathlib import Path
from transformers import BertTokenizerFast, BertForMaskedLM

def parse_args():
    ap = argparse.ArgumentParser(
        "Evaluate a LOCAL BERT (Masked LM) on fill-in-the-blank prompts."
    )
    ap.add_argument("model_dir", type=str, help="path to local model folder (no HF hub)")
    ap.add_argument("--eval_file", type=str, default="data/eval.txt",
                    help="file with one prompt per line, each containing [MASK]")
    ap.add_argument("--topk", type=int, default=5, help="how many predictions to print")
    ap.add_argument("--pool", type=int, default=200,
                    help="candidate pool depth before filtering (higher = slower but better)")
    ap.add_argument("--raw", action="store_true",
                    help="print raw top-k (no filtering); includes punctuation/subwords/stopwords")
    ap.add_argument("--show_probs", action="store_true",
                    help="print probabilities next to predictions")
    return ap.parse_args()

def main():
    args = parse_args()

    model_path = Path(args.model_dir).expanduser().resolve()
    eval_file  = Path(args.eval_file).expanduser().resolve()

    if not model_path.is_dir():
        print(f"[ERROR] '{model_path}' is not a local model directory."); sys.exit(1)
    if not eval_file.exists():
        print(f"[ERROR] '{eval_file}' not found."); sys.exit(1)

    print(f"[INFO] Loading LOCAL model from {model_path}")
    tok = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)
    model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
    model.eval()

    lines = [l.strip() for l in eval_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[INFO] Loaded {len(lines)} prompts from {eval_file}")

    # Filtering helpers (used unless --raw is set)
    alpha = re.compile(r"^[A-Za-z]+$")
    STOPWORDS = {
        "the","a","an","to","of","in","on","at","for","and","or","but","with","by","from",
        "as","is","are","was","were","be","been","being","it","that","this","these","those",
        "he","she","they","we","you","i","his","her","their","our","your","my","do","does",
        "did","have","has","had","will","would","can","could","should","may","might","must",
        "s","t","m","re","ve","ll"
    }
    def is_clean_word(tok_str: str) -> bool:
        if tok_str in tok.all_special_tokens: return False
        if tok_str.startswith("##"):          return False   # subword continuation
        if not alpha.match(tok_str):          return False   # punctuation/digits
        if tok_str.lower() in STOPWORDS:      return False   # function words
        return True

    for prompt in lines:
        enc = tok(prompt, return_tensors="pt")
        mask_pos = (enc.input_ids == tok.mask_token_id)[0].nonzero(as_tuple=True)[0]
        if mask_pos.numel() == 0:
            print(f"\n[WARN] No [MASK] token in prompt: {prompt}")
            continue
        if mask_pos.numel() > 1:
            print(f"\n[WARN] Multiple [MASK] found; using the first one.")

        pos = mask_pos[0].item()
        with torch.no_grad():
            logits = model(**enc).logits[0, pos, :]
        probs = torch.softmax(logits, dim=-1)

        print(f"\nPrompt: {prompt}")

        if args.raw:
            # Raw top-k from full vocab (may include punctuation/subwords/stopwords)
            vals, idxs = torch.topk(probs, args.topk)
            for i, (tid, p) in enumerate(zip(idxs.tolist(), vals.tolist()), 1):
                w = tok.convert_ids_to_tokens(tid)
                if args.show_probs:
                    print(f"  {i:>2}. {w:>15} ({p:.3f})")
                else:
                    print(f"  {i:>2}. {w}")
            continue

        # Filtered: take a deeper pool, then keep the first K clean word tokens
        vals, idxs = torch.topk(probs, max(args.pool, args.topk))
        kept = []
        for tid, p in zip(idxs.tolist(), vals.tolist()):
            w = tok.convert_ids_to_tokens(tid)
            if is_clean_word(w):
                kept.append((w, p))
                if len(kept) == args.topk:
                    break

        if kept:
            for i, (w, p) in enumerate(kept, 1):
                if args.show_probs:
                    print(f"  {i:>2}. {w:>15} ({p:.3f})")
                else:
                    print(f"  {i:>2}. {w}")
        else:
            # fall back to raw if nothing survived filtering
            print("  (no clean word tokens in top pool; showing raw top-k)")
            vals, idxs = torch.topk(probs, args.topk)
            for i, (tid, p) in enumerate(zip(idxs.tolist(), vals.tolist()), 1):
                w = tok.convert_ids_to_tokens(tid)
                if args.show_probs:
                    print(f"  {i:>2}. {w:>15} ({p:.3f})")
                else:
                    print(f"  {i:>2}. {w}")

if __name__ == "__main__":
    main()
