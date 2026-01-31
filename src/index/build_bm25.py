import os
import json
import pickle
import argparse
from tqdm import tqdm

def tokenize(text: str):
    return text.lower().split()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="Path to docs.jsonl")
    ap.add_argument("--out", required=True, help="Output path for bm25 index (pkl)")
    args = ap.parse_args()

    from rank_bm25 import BM25Okapi

    doc_ids = []
    tokenized_corpus = []

    with open(args.docs, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading docs"):
            obj = json.loads(line)
            doc_id = obj["doc_id"]
            text = (obj.get("title", "") + "\n" + obj.get("text", "")).strip()
            doc_ids.append(doc_id)
            tokenized_corpus.append(tokenize(text))

    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"doc_ids": doc_ids, "bm25": bm25}, f)

    print(f"Saved BM25 index to: {args.out}")
    print(f"Docs indexed: {len(doc_ids)}")

if __name__ == "__main__":
    main()
