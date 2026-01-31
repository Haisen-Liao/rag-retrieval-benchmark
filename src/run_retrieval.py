import os
import json
import time
import argparse
import yaml
from tqdm import tqdm


def load_queries(path):
    """Load queries.jsonl -> List[(qid, query)]"""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries.append((obj["qid"], obj["query"]))
    return queries


def load_doc_texts(docs_path, max_doc_chars=None):
    """
    Load docs.jsonl -> dict[doc_id] = "title\\ntext"
    Optionally truncate to max_doc_chars for speed/stability.
    """
    doc_map = {}
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj["doc_id"]
            text = (obj.get("title", "") + "\n" + obj.get("text", "")).strip()
            if max_doc_chars is not None:
                text = text[: int(max_doc_chars)]
            doc_map[doc_id] = text
    return doc_map


def build_retriever(r_cfg: dict):
    r_type = r_cfg["type"].lower()

    if r_type == "bm25":
        from src.retrieve.bm25_retriever import BM25Retriever
        return BM25Retriever(r_cfg["bm25_index_path"])

    if r_type == "dense":
        from src.retrieve.dense_retriever import DenseRetriever
        return DenseRetriever(
            index_path=r_cfg["faiss_index_path"],
            store_path=r_cfg["store_path"],
            model_name=r_cfg["embedding_model"],
        )

    if r_type == "hybrid":
        from src.retrieve.dense_retriever import DenseRetriever
        from src.retrieve.bm25_retriever import BM25Retriever
        from src.retrieve.hybrid_retriever import HybridRetriever

        dense = DenseRetriever(
            index_path=r_cfg["faiss_index_path"],
            store_path=r_cfg["store_path"],
            model_name=r_cfg["embedding_model"],
        )
        bm25 = BM25Retriever(r_cfg["bm25_index_path"])

        return HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            alpha=float(r_cfg.get("alpha", 0.5)),
        )

    if r_type == "rrf":
        from src.retrieve.dense_retriever import DenseRetriever
        from src.retrieve.bm25_retriever import BM25Retriever
        from src.retrieve.rrf_retriever import RRFRetriever

        dense = DenseRetriever(
            index_path=r_cfg["faiss_index_path"],
            store_path=r_cfg["store_path"],
            model_name=r_cfg["embedding_model"],
        )
        bm25 = BM25Retriever(r_cfg["bm25_index_path"])

        return RRFRetriever(
            retrievers=[dense, bm25],
            k=int(r_cfg.get("rrf_k", 60)),
        )

    raise ValueError(f"Unsupported retrieval.type: {r_type}")


def build_reranker(rerank_cfg: dict):
    """
    Cross-encoder reranker.
    Expects src/rerank/cross_encoder_reranker.py to exist.
    """
    from src.rerank.cross_encoder_reranker import CrossEncoderReranker
    return CrossEncoderReranker(
        model_name=rerank_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        batch_size=int(rerank_cfg.get("batch_size", 32)),
    )


def minmax_norm(score_map: dict):
    """
    score_map: dict[str,float]
    return dict[str,float] normalized to [0,1]
    """
    if not score_map:
        return {}
    vals = list(score_map.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-12:
        return {k: 1.0 for k in score_map.keys()}
    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to method yaml config")
    ap.add_argument("--queries", required=True, help="Path to queries.jsonl")
    ap.add_argument("--out", required=True, help="Output run jsonl path")
    args = ap.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    r_cfg = cfg["retrieval"]
    r_type = r_cfg["type"].lower()

    # Retrieval settings
    retrieval_top_k = int(r_cfg.get("top_k", 100))

    # Rerank settings
    rerank_cfg = cfg.get("rerank", {}) or {}
    rerank_enabled = bool(rerank_cfg.get("enabled", False))

    # Build retriever
    retriever = build_retriever(r_cfg)

    # If reranking: load docs and build reranker
    doc_texts = None
    reranker = None

    # rerank parameters (only valid if rerank_enabled)
    cand_k = None
    out_k = None
    rerank_mode = None  # "hard" or "fusion"
    lam = None
    use_minmax = None

    if rerank_enabled:
        docs_path = rerank_cfg["docs_path"]
        max_doc_chars = rerank_cfg.get("max_doc_chars", None)  # e.g. 1200
        doc_texts = load_doc_texts(docs_path, max_doc_chars=max_doc_chars)

        reranker = build_reranker(rerank_cfg)

        cand_k = int(rerank_cfg.get("candidate_k", 20))
        out_k = int(rerank_cfg.get("top_k", 100))

        # mode: "fusion" recommended, "hard" = pure rerank takeover
        rerank_mode = str(rerank_cfg.get("mode", "fusion")).lower()
        if rerank_mode not in ("fusion", "hard"):
            raise ValueError("rerank.mode must be one of: fusion, hard")

        lam = float(rerank_cfg.get("lambda", 0.2))  # only for fusion
        use_minmax = bool(rerank_cfg.get("minmax_norm", True))

        if retrieval_top_k < cand_k:
            raise ValueError(
                f"retrieval.top_k ({retrieval_top_k}) must be >= rerank.candidate_k ({cand_k})"
            )
        if out_k > retrieval_top_k:
            # not strictly required, but keeps output bounded by retrieved candidates
            # you can relax this if you want, but usually retrieval_top_k>=out_k is expected
            raise ValueError(
                f"rerank.top_k ({out_k}) must be <= retrieval.top_k ({retrieval_top_k})"
            )

    # Load queries
    queries = load_queries(args.queries)

    # Ensure output dir exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    with open(args.out, "w", encoding="utf-8") as fout:
        for qid, q in tqdm(
            queries,
            desc=f"Retrieving ({r_type}{'+rerank' if rerank_enabled else ''})",
        ):
            # 1) retrieve candidates
            results = retriever.search(q, top_k=retrieval_top_k)

            # 2) optional rerank
            if rerank_enabled:
                cand = results[:cand_k]

                # Prepare rerank inputs + keep retrieval scores for fusion
                docs_for_rerank = []
                ret_score_map = {}
                for r in cand:
                    doc_id = r["doc_id"]
                    text = doc_texts.get(doc_id, "")
                    docs_for_rerank.append((doc_id, text))
                    ret_score_map[doc_id] = float(r.get("score", 0.0))

                # Cross-encoder scores for cand
                reranked = reranker.rerank(q, docs_for_rerank, top_k=cand_k)
                rr_score_map = {x["doc_id"]: float(x["score"]) for x in reranked}

                if rerank_mode == "hard":
                    # Pure rerank takeover within cand, then append the rest
                    reranked_ids = {r["doc_id"] for r in reranked}
                    rest = [r for r in results if r["doc_id"] not in reranked_ids]
                    results = (reranked + rest)[:out_k]

                else:
                    # ---- Fusion rerank (interpolated) ----
                    if use_minmax:
                        ret_norm = minmax_norm(ret_score_map)
                        rr_norm = minmax_norm(rr_score_map)
                    else:
                        ret_norm = ret_score_map
                        rr_norm = rr_score_map

                    fused = []
                    for doc_id in ret_score_map.keys():
                        s_ret = ret_norm.get(doc_id, 0.0)
                        s_rr = rr_norm.get(doc_id, 0.0)
                        s = (1.0 - lam) * s_ret + lam * s_rr
                        fused.append({"doc_id": doc_id, "score": float(s)})

                    fused.sort(key=lambda x: x["score"], reverse=True)

                    fused_ids = {r["doc_id"] for r in fused}
                    rest = [r for r in results if r["doc_id"] not in fused_ids]

                    results = (fused + rest)[:out_k]
                    # ---- end Fusion rerank ----

            # 3) write run line (always)
            fout.write(json.dumps({"qid": qid, "results": results}, ensure_ascii=False) + "\n")

    t1 = time.time()

    print(f"Saved run to: {args.out}")
    print(f"Total queries: {len(queries)}")
    print(f"Elapsed: {t1 - t0:.2f}s")
    if rerank_enabled:
        print(
            f"Rerank enabled: mode={rerank_mode}, candidate_k={cand_k}, out_k={out_k}, "
            f"lambda={lam if rerank_mode=='fusion' else 'N/A'}, "
            f"minmax_norm={use_minmax}, max_doc_chars={rerank_cfg.get('max_doc_chars', None)}"
        )


if __name__ == "__main__":
    main()

