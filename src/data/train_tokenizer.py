# src/data/train_tokenizer.py
import json
from pathlib import Path
from tokenizers import BertWordPieceTokenizer

OUT = Path("tokenizer")
OUT.mkdir(exist_ok=True, parents=True)

# Train on your tiny corpus (one doc per line)
corpus_path = Path("data/corpus_original.txt")
assert corpus_path.exists(), f"Missing {corpus_path}"

tok = BertWordPieceTokenizer(lowercase=True, strip_accents=True)
tok.train(files=[str(corpus_path)], vocab_size=8000,
          special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"])

# Save vocab.txt
tok.save_model(str(OUT))

# Write the minimal config so Transformers can load it locally
(OUT / "tokenizer_config.json").write_text(json.dumps({
    "do_lower_case": True
}, indent=2))

(OUT / "special_tokens_map.json").write_text(json.dumps({
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]"
}, indent=2))

print("Tokenizer saved to ./tokenizer (vocab.txt + configs)")

