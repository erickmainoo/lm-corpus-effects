#!/usr/bin/env python3
import argparse, json, time, random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)  # type: ignore
        except Exception:
            pass

def read_lines(path: Path, limit: int | None = None) -> list[str]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
                if limit and len(out) >= limit:
                    break
    return out

def read_qa_pairs(path: Path):
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                continue
            prompt, ans = line.split("\t", 1)
            prompt = prompt.strip()
            ans = ans.strip()
            if "[MASK]" not in prompt:
                continue
            if prompt and ans:
                pairs.append((prompt, ans))
    return pairs

def make_hf_dataset(lines: list[str], tok, max_len: int) -> Dataset:
    ds = Dataset.from_dict({"text": lines})
    return ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=max_len),
        batched=True,
        remove_columns=["text"],
    )

class EpochTimer(TrainerCallback):
    def __init__(self):
        self.epoch_times = []
        self._t0 = None
    def on_epoch_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()
    def on_epoch_end(self, args, state, control, **kwargs):
        if self._t0 is not None:
            self.epoch_times.append(time.perf_counter() - self._t0)
        return control

class EpochQAEval(TrainerCallback):
    def __init__(self, tokenizer, qa_pairs, device):
        self.tok = tokenizer
        self.qa_pairs = qa_pairs
        self.device = device
        self.results = []  # list of dicts per epoch

    @staticmethod
    def _norm_token(tok: str) -> str:
        tok = tok.lower()
        if tok.startswith("##"):
            tok = tok[2:]
        return tok.strip(" .,!?\"'")

    @staticmethod
    def _norm_answer(ans: str) -> str:
        return ans.lower().strip(" .,!?\"'")

    def _eval_once(self, model):
        model.eval()
        correct = 0
        total = len(self.qa_pairs)
        samples = []
        for prompt, gold in self.qa_pairs:
            inputs = self.tok(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits
            mask_pos = (inputs["input_ids"] == self.tok.mask_token_id).nonzero(as_tuple=True)
            if len(mask_pos[1]) == 0:
                continue
            scores = logits[mask_pos][0]
            probs = torch.softmax(scores, dim=-1)
            pred_id = probs.argmax(-1).item()
            pred_token = self.tok.convert_ids_to_tokens([pred_id])[0]

            pred_norm = self._norm_token(pred_token)
            gold_norm = self._norm_answer(gold)

            is_correct = int(pred_norm == gold_norm)
            correct += is_correct

            if len(samples) < 10:  # keep a few examples
                samples.append({
                    "prompt": prompt,
                    "gold": gold,
                    "pred_raw": pred_token,
                    "pred_norm": pred_norm,
                    "correct": bool(is_correct),
                })
        acc = correct / total if total > 0 else 0.0
        return acc, correct, total, samples

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or not self.qa_pairs:
            return control
        acc, correct, total, samples = self._eval_once(model)
        epoch_num = state.epoch
        print(f"\n[QA-EVAL] epoch={epoch_num:.1f} acc={acc:.3f} ({correct}/{total})\n")
        self.results.append({
            "epoch": epoch_num,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "sample_predictions": samples,
        })
        return control

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Training corpus (one doc per line)")
    ap.add_argument("--qa_file", required=True, help="TSV with 'prompt[TAB]answer'")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_lines = read_lines(Path(args.corpus), args.limit_train)
    qa_pairs = read_qa_pairs(Path(args.qa_file))

    if not train_lines:
        raise SystemExit("no training lines")
    if not qa_pairs:
        raise SystemExit("no QA pairs")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "[MASK]"})

    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tok))
    model.to(device)

    train_ds = make_hf_dataset(train_lines, tok, args.max_len)

    # use QA prompts as a tiny eval set for loss
    eval_lines = [p for (p, _) in qa_pairs]
    eval_ds = make_hf_dataset(eval_lines, tok, args.max_len)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm_probability=args.mlm_prob)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        logging_steps=50,
    )

    timer_cb = EpochTimer()
    qa_cb = EpochQAEval(tok, qa_pairs, device)

    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        callbacks=[timer_cb, qa_cb],
    )

    t0 = time.perf_counter()
    trainer.train()
    total_time = time.perf_counter() - t0

    metrics = trainer.evaluate()
    eval_loss = float(metrics.get("eval_loss", 0.0))

    approx_tokens = int(sum(len(x) for x in train_ds["input_ids"]))
    tokens_per_sec = round(approx_tokens / total_time, 1) if total_time > 0 else None

    out_json = {
        "model": args.model,
        "device": device,
        "train_lines": len(train_lines),
        "epochs": args.epochs,
        "batch_size": args.bsz,
        "learning_rate": args.lr,
        "mlm_probability": args.mlm_prob,
        "max_len": args.max_len,
        "epoch_times_sec": [round(t, 2) for t in timer_cb.epoch_times],
        "total_time_sec": round(total_time, 2),
        "eval_loss": eval_loss,
        "approx_tokens_trained": approx_tokens,
        "tokens_per_sec": tokens_per_sec,
        "qa_epoch_results": qa_cb.results,
    }

    out_path = out_dir / "run_metrics.json"
    out_path.write_text(json.dumps(out_json, indent=2))
    print(json.dumps(out_json, indent=2))

if __name__ == "__main__":
    main()
