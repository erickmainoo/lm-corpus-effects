#!/usr/bin/env python3
import json, csv
from pathlib import Path

rows = []
for d in sorted(Path("runs").glob("*")):
    p = d / "run_metrics.json"
    if p.exists():
        r = json.loads(p.read_text())
        rows.append({
            "run": d.name,
            "train_lines": r.get("train_lines"),
            "epochs": r.get("epochs"),
            "bsz": r.get("batch_size"),
            "total_time_sec": r.get("total_time_sec"),
            "epoch_time_sec": (r.get("epoch_times_sec") or [None])[0],
            "eval_loss": r.get("eval_loss"),
            "tokens_per_sec": r.get("tokens_per_sec"),
        })

rows.sort(key=lambda x: (("filtered" in x["run"]), x["train_lines"]))  # originals first, then filtered

out_csv = Path("runs/summary.csv")
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"wrote {out_csv}")
for r in rows:
    print(f"{r['run']:>14} | lines={r['train_lines']:>5} | epoch={r['epoch_time_sec']:>6} s | total={r['total_time_sec']:>6} s | loss={r['eval_loss']:.3f} | tok/s={r['tokens_per_sec']}")
