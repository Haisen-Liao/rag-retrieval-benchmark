import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="docs.jsonl")
    ap.add_argument("--index_out", required=True, help="FAISS index path")
    ap.add_argument("--store_out", required=True, help="doc store jsonl path")
    ap.add_argument("--model_name", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    model = SentenceTransformer(args.model_name)

    doc_ids = []
    texts = []

    with open(args.docs, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj["doc_id"]
            text = (obj.get("title", "") + "\n" + obj.get("text", "")).strip()
            doc_ids.append(doc_id)
            texts.append(text)

    # Encode
    embeddings = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Encoding docs"):
        batch = texts[i:i+args.batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(emb)

    emb = np.vstack(embeddings).astype("float32")
    dim = emb.shape[1]

    # Build FAISS (cosine via inner product because we normalized embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)
    faiss.write_index(index, args.index_out)

    # Save store
    os.makedirs(os.path.dirname(args.store_out), exist_ok=True)
    with open(args.store_out, "w", encoding="utf-8") as fout:
        for doc_id, text in zip(doc_ids, texts):
            fout.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")

    print(f"Saved FAISS index to: {args.index_out}")
    print(f"Saved store to: {args.store_out}")
    print(f"Docs indexed: {len(doc_ids)}")

if __name__ == "__main__":
    main()
