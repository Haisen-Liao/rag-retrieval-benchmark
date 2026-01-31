from typing import List, Tuple
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32):
        self.model = CrossEncoder(model_name)
        self.batch_size = int(batch_size)

    def rerank(self, query: str, docs: List[Tuple[str, str]], top_k: int):
        """
        docs: list of (doc_id, doc_text)
        return: list of {"doc_id":..., "score":...} sorted desc
        """
        pairs = [[query, text] for _, text in docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        scored = [{"doc_id": doc_id, "score": float(s)} for (doc_id, _), s in zip(docs, scores)]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
