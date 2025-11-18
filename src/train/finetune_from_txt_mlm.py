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
    TrainerCallback,   # <-- important
)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try: torch.mps.manual_seed(seed)  # type: ignore
        except Exception: pass

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
        return control  # older versions expect returning control

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_eval", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_lines = read_lines(Path(args.corpus), args.limit_train)
    eval_lines  = read_lines(Path(args.eval), args.limit_eval)
    if not train_lines: raise SystemExit("no training lines")
    if not eval_lines:  raise SystemExit("no eval lines")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "[MASK]"})

    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tok))
    model.to(device)

    train_ds = make_hf_dataset(train_lines, tok, args.max_len)
    eval_ds  = make_hf_dataset(eval_lines, tok, args.max_len)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm_probability=args.mlm_prob)

    # Use only arguments supported by older transformers versions
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        logging_steps=50,
    )

    timer_cb = EpochTimer()
    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,  # FutureWarning is fine; safe to ignore
        callbacks=[timer_cb],
    )

    t0 = time.perf_counter()
    if args.epochs > 0:
        trainer.train()
    total_time = time.perf_counter() - t0

    # Run a single eval pass (older versions won’t auto-eval each epoch with minimal args)
    metrics = trainer.evaluate()
    eval_loss = float(metrics.get("eval_loss", 0.0))

    approx_tokens = int(sum(len(x) for x in train_ds["input_ids"]))
    tokens_per_sec = round(approx_tokens / total_time, 1) if total_time > 0 else None

    def fill_mask(prompt: str, topk: int = 5):
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        mask_pos = (inputs["input_ids"] == tok.mask_token_id).nonzero(as_tuple=True)
        if len(mask_pos[1]) == 0: return []
        scores = logits[mask_pos][0].softmax(-1)
        ids = torch.topk(scores, k=topk).indices.tolist()
        return tok.convert_ids_to_tokens(ids)

    sample_prompts = [l for l in eval_lines if "[MASK]" in l][:5]
    sample_fills = {p: fill_mask(p) for p in sample_prompts}

    out_json = {
        "model": args.model,
        "device": device,
        "train_lines": len(train_lines),
        "eval_lines": len(eval_lines),
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
        "samples": sample_fills,
    }
    (out_dir / "run_metrics.json").write_text(json.dumps(out_json, indent=2))
    print(json.dumps(out_json, indent=2))

    if args.save_model:
        (out_dir / "model").mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir / "model")
        tok.save_pretrained(out_dir / "model")
        print(f"[INFO] model saved to {out_dir / 'model'}")

if __name__ == "__main__":
    main()
