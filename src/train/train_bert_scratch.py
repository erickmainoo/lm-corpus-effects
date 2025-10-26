# src/train/train_bert_scratch.py
import argparse, math, random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
)

def load_corpus(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus not found: {p}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()]
    return [d for d in lines if d]

def build_datasets(docs, tokenizer, max_length=128, seed=42, val_ratio=0.1):
    random.Random(seed).shuffle(docs)
    n_val = max(1, int(len(docs) * val_ratio))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds_train = Dataset.from_dict({"text": train_docs}).map(tok, batched=True, remove_columns=["text"])
    ds_val   = Dataset.from_dict({"text": val_docs}).map(tok, batched=True, remove_columns=["text"])
    return ds_train, ds_val

def main():
    ap = argparse.ArgumentParser("Train a tiny BERT (from scratch) on a small corpus for MLM")
    ap.add_argument("corpus_path", type=str, help="path to corpus .txt (one doc per line)")
    ap.add_argument("output_dir", type=str, help="where to save the randomly-initialized trained model")

    # Training hyperparams (tuned for tiny-from-scratch)
    ap.add_argument("--epochs", type=float, default=20.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")

    # Tiny model size (fast on CPU; bump on GPU if desired)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--intermediate_size", type=int, default=256)
    args = ap.parse_args()

    set_seed(args.seed)

    # 1) Tokenizer: reuse bert-base-uncased vocab (OK for from-scratch weights)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # 2) Random-init tiny BERT config (this is the “from scratch” part)
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=512,
        type_vocab_size=2,
    )
    model = BertForMaskedLM(cfg)

    # 3) Data
    docs = load_corpus(args.corpus_path)
    ds_train, ds_val = build_datasets(docs, tokenizer, max_length=args.max_length, seed=args.seed)
    collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)

    # 4) Training args (new + fallback for older transformers)
    try:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.06,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=50,
            fp16=args.fp16 and torch.cuda.is_available(),
            report_to=[],
            dataloader_num_workers=0,
        )
    except TypeError:
        # Older transformers: no evaluation_strategy/save_strategy/warmup_ratio
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=0.01,
            logging_steps=50,
            fp16=args.fp16 and torch.cuda.is_available(),
            dataloader_num_workers=0,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print(f"[INFO] Training from scratch on {len(ds_train)} train / {len(ds_val)} val docs …")
    trainer.train()

    print("[INFO] Evaluating …")
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss")
    ppl = math.exp(eval_loss) if (eval_loss is not None and eval_loss < 20) else float("inf")
    print(f"[RESULT] eval_loss={eval_loss:.4f} | perplexity={ppl:.2f}")

    # 5) Save model + tokenizer + metadata
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "scratch_metadata.txt").write_text(
        f"corpus_path={Path(args.corpus_path).resolve()}\n"
        f"epochs={args.epochs}\n"
        f"batch_size={args.batch_size}\n"
        f"grad_accum={args.grad_accum}\n"
        f"lr={args.lr}\n"
        f"hidden_size={args.hidden_size}\n"
        f"layers={args.layers}\n"
        f"heads={args.heads}\n"
        f"intermediate_size={args.intermediate_size}\n"
        f"max_length={args.max_length}\n"
        f"mlm_prob={args.mlm_prob}\n"
        f"seed={args.seed}\n",
        encoding="utf-8",
    )
    print(f"[DONE] Saved to {out.resolve()}")

if __name__ == "__main__":
    main()
