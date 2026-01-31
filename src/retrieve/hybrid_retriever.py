from collections import defaultdict

class HybridRetriever:
    """
    Combine dense and bm25 results with min-max normalized scores:
      score = alpha * dense_norm + (1 - alpha) * bm25_norm
    """
    def __init__(self, dense_retriever, bm25_retriever, alpha: float = 0.5):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.alpha = float(alpha)

    @staticmethod
    def _minmax_norm(results):
        # results: [{"doc_id":..., "score":...}, ...]
        if not results:
            return {}
        scores = [r["score"] for r in results]
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-12:
            return {r["doc_id"]: 1.0 for r in results}  # all same
        return {r["doc_id"]: (r["score"] - mn) / (mx - mn) for r in results}

    def search(self, query: str, top_k: int = 100, dense_k: int = 200, bm25_k: int = 200):
        dense_res = self.dense.search(query, top_k=dense_k)
        bm25_res = self.bm25.search(query, top_k=bm25_k)

        dense_norm = self._minmax_norm(dense_res)
        bm25_norm = self._minmax_norm(bm25_res)

        # union of doc_ids
        all_docs = set(dense_norm.keys()) | set(bm25_norm.keys())

        combined = []
        for doc_id in all_docs:
            s_dense = dense_norm.get(doc_id, 0.0)
            s_bm25 = bm25_norm.get(doc_id, 0.0)
            s = self.alpha * s_dense + (1.0 - self.alpha) * s_bm25
            combined.append({"doc_id": doc_id, "score": float(s)})

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]
