import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, index_path: str, store_path: str, model_name: str):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(model_name)

        self.doc_ids = []
        with open(store_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.doc_ids.append(obj["doc_id"])

    def search(self, query: str, top_k: int = 100):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, idx = self.index.search(q_emb, top_k)

        results = []
        for i, s in zip(idx[0], scores[0]):
            results.append({"doc_id": self.doc_ids[int(i)], "score": float(s)})
        return results
