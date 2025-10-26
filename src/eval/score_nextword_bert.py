import sys, math, torch
from pathlib import Path
from transformers import BertTokenizerFast, BertForMaskedLM

USAGE = "Usage: python src/eval/score_nextword_bert.py <local_model_dir> <prompts.txt> <answers.tsv>"

def load_lines(p: Path):
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(USAGE); sys.exit(1)
    model_dir = Path(sys.argv[1])
    prompts_p = Path(sys.argv[2])
    answers_p = Path(sys.argv[3])

    if not model_dir.is_dir():
        print(f"[ERROR] {model_dir} is not a local model folder"); sys.exit(1)
    if not prompts_p.exists() or not answers_p.exists():
        print(f"[ERROR] Missing prompts or answers file"); sys.exit(1)

    tok = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    model = BertForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    prompts = load_lines(prompts_p)
    golds = load_lines(answers_p)
    assert len(prompts) == len(golds), "Prompts and answers must align 1:1"

    n = len(prompts)
    acc1 = acc5 = 0
    gold_probs = []

    for prompt, gold in zip(prompts, golds):
        enc = tok(prompt, return_tensors="pt")
        # find [MASK] position (we expect exactly 1)
        mask_pos = (enc.input_ids == tok.mask_token_id)[0].nonzero(as_tuple=True)[0]
        if mask_pos.numel() == 0:
            continue
        pos = mask_pos[0].item()

        with torch.no_grad():
            logits = model(**enc).logits  # [1, L, V]
        logits_row = logits[0, pos, :]   # [V]
        probs = torch.softmax(logits_row, dim=-1)  # full vocab (no filters)

        # gold may tokenize into multiple pieces; for "next word" we score the FIRST piece
        gold_pieces = tok.tokenize(gold)
        if not gold_pieces:
            # unknown: score as 0 prob and continue
            gold_id = tok.unk_token_id
        else:
            gold_id = tok.convert_tokens_to_ids(gold_pieces[0])

        vals, idxs = torch.topk(probs, 5)
        top_ids = idxs.tolist()
        acc1 += int(top_ids[0] == gold_id)
        acc5 += int(gold_id in top_ids)
        gold_probs.append(probs[gold_id].item() if gold_id is not None else 0.0)

    acc1 /= n
    acc5 /= n
    avg_gold = sum(gold_probs)/len(gold_probs) if gold_probs else float("nan")

    print(f"[RESULT] n={n}  acc@1={acc1:.3f}  acc@5={acc5:.3f}  avg_gold_prob={avg_gold:.4f}")
