class RRFRetriever:
    """
    Reciprocal Rank Fusion:
      score(doc) = sum( 1 / (k + rank_i(doc)) ) over systems i
    k is usually 60.
    """
    def __init__(self, retrievers, k: int = 60):
        self.retrievers = retrievers
        self.k = int(k)

    def search(self, query: str, top_k: int = 100, per_system_k: int = 200):
        scores = {}

        for r in self.retrievers:
            res = r.search(query, top_k=per_system_k)
            for rank, item in enumerate(res, start=1):
                doc_id = item["doc_id"]
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"doc_id": d, "score": float(s)} for d, s in ranked]
