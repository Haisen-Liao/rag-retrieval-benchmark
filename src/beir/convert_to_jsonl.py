import os
import json
import csv
import argparse
from tqdm import tqdm


def convert_docs(raw_dir, out_dir, dataset):
    os.makedirs(out_dir, exist_ok=True)
    in_path = os.path.join(raw_dir, "corpus.jsonl")
    out_path = os.path.join(out_dir, "docs.jsonl")

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"[{dataset}] Converting docs"):
            obj = json.loads(line)
            out = {
                "doc_id": obj["_id"],
                "title": obj.get("title", ""),
                "text": obj.get("text", ""),
                "metadata": {
                    "source": f"beir-{dataset}"
                }
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


def convert_queries(raw_dir, out_dir, dataset):
    in_path = os.path.join(raw_dir, "queries.jsonl")
    out_path = os.path.join(out_dir, "queries.jsonl")

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"[{dataset}] Converting queries"):
            obj = json.loads(line)
            out = {
                "qid": obj["_id"],
                "query": obj["text"]
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


def convert_qrels(raw_dir, out_dir, dataset):
    qrels_dir = os.path.join(raw_dir, "qrels")

    for split in ["test.tsv", "dev.tsv"]:
        path = os.path.join(qrels_dir, split)
        if os.path.exists(path):
            qrels_file = path
            break
    else:
        raise FileNotFoundError(f"No qrels file found in {qrels_dir}")

    out_path = os.path.join(out_dir, "qrels.jsonl")

    with open(qrels_file, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in tqdm(reader, desc=f"[{dataset}] Converting qrels"):
            out = {
                "qid": row["query-id"],
                "doc_id": row["corpus-id"],
                "relevance": int(row["score"])
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="BEIR dataset name, e.g. scifact, trec-covid")
    parser.add_argument("--raw_root", type=str, default="data/beir_raw")
    parser.add_argument("--out_root", type=str, default="data/processed")
    args = parser.parse_args()

    raw_dir = os.path.join(args.raw_root, args.dataset)
    out_dir = os.path.join(args.out_root, args.dataset)

    print(f"Converting BEIR dataset [{args.dataset}] ...")
    convert_docs(raw_dir, out_dir, args.dataset)
    convert_queries(raw_dir, out_dir, args.dataset)
    convert_qrels(raw_dir, out_dir, args.dataset)
    print("Done.")
