import time, math, argparse, os, random
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoConfig, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)

def load_ag_news_subsets(size:int, withheld_label:int|None):
    ds = load_dataset("ag_news")
    # concat train+test, shuffle, then sample "size" examples
    full = ds["train"].shuffle(seed=42).select(range(min(size*2, len(ds["train"]))))
    # optional filtering to emulate "current with one topic withheld"
    if withheld_label is not None:
        full = full.filter(lambda ex: ex["label"] != withheld_label)
    # take first "size" for train, next ~10% for eval
    take = min(size, len(full))
    train = full.select(range(take))
    eval_size = max(100, take//10)
    eval_end = min(take+eval_size, len(full))
    evald = full.select(range(take, eval_end))
    return DatasetDict(train=train, validation=evald)

def build_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.mask_token is None:  # safety for some checkpoints
        tok.add_special_tokens({"mask_token": "[MASK]"})
    return tok

def tokenize_fn(tok, max_len=128):
    def _f(batch): return tok(batch["text"], truncation=True, max_length=max_len)
    return _f

class EpochTimerCallback:
    def __init__(self): self.epoch_times=[]; self._t=None
    def on_epoch_begin(self, args, state, control, **kwargs): self._t=time.perf_counter()
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_times.append(time.perf_counter()-self._t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--size", type=int, default=1000, help="num training docs")
    ap.add_argument("--withhold_label", type=int, default=None, help="0=World,1=Sports,2=Business,3=Sci/Tech")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default="out_bert_mlm")
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    set_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # data
    dsd = load_ag_news_subsets(args.size, args.withhold_label)
    tok = build_tokenizer(args.model)
    dsd = dsd.map(tokenize_fn(tok, args.max_len), batched=True, remove_columns=dsd["train"].column_names)

    # model
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, config=cfg)
    # BERT base already has [MASK]; just in case we added, resize embeddings:
    model.resize_token_embeddings(len(tok))
    if device=="mps": model.to("mps")

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm_probability=args.mlm_prob)

    # training args
    out_dir = f"{args.out}_size{args.size}" + (f"_no{args.withhold_label}" if args.withhold_label is not None else "")
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none"
    )

    timer_cb = EpochTimerCallback()
    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=collator,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tok,
        callbacks=[timer_cb]
    )

    # Train + time
    t0 = time.perf_counter()
    trainer.train()
    total_time = time.perf_counter()-t0

    # Eval loss as a quick proxy
    metrics = trainer.evaluate()
    # Throughput
    num_tokens = sum(len(x) for x in dsd["train"]["input_ids"])
    tok_per_sec = num_tokens / total_time if total_time>0 else float("nan")

    print("\n=== TIMING / METRICS ===")
    print(f"epoch_times_sec: {timer_cb.epoch_times}")
    print(f"total_time_sec: {total_time:.2f}")
    print(f"eval_loss: {metrics.get('eval_loss'):.4f}")
    print(f"approx_tokens_trained: {num_tokens}")
    print(f"tokens_per_sec: {tok_per_sec:.1f}")

    # quick fill-in-the-blank demo
    def fill_blank(prompt_with_mask:str, topk=5):
        inputs = tok(prompt_with_mask, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        mask_idx = (inputs["input_ids"] == tok.mask_token_id).nonzero(as_tuple=True)
        last_dim = logits[mask_idx][0]  # logits for the [MASK] position(s)
        probs = last_dim.softmax(-1)
        topk_ids = torch.topk(probs, k=topk).indices.tolist()
        return tok.convert_ids_to_tokens(topk_ids)

    print("\nFill-in-the-blank examples:")
    for p in [
        "The economy is showing signs of [MASK].",
        "The team secured a decisive [MASK] in the final.",
        "Scientists discovered a new [MASK] for the treatment."
    ]:
        print(p, "->", fill_blank(p))

if __name__ == "__main__":
    main()
