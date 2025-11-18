import argparse
from datasets import Dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    a = p.parse_args()
    lines = [l.strip() for l in open(a.inp, encoding="utf-8") if l.strip()]
    Dataset.from_dict({"text": lines}).save_to_disk(a.out)
    print(f"Saved {len(lines)} docs to {a.out}")

if __name__ == "__main__":
    main()
