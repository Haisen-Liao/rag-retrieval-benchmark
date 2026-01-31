import pickle
import numpy as np

def tokenize(text: str):
    return text.lower().split()

class BM25Retriever:
    def __init__(self, bm25_pkl_path: str):
        with open(bm25_pkl_path, "rb") as f:
            payload = pickle.load(f)
        self.doc_ids = payload["doc_ids"]
        self.bm25 = payload["bm25"]

    def search(self, query: str, top_k: int = 100):
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        scores = np.asarray(scores)

        top_idx = np.argsort(-scores)[:top_k]
        results = [{"doc_id": self.doc_ids[i], "score": float(scores[i])} for i in top_idx]
        return results
