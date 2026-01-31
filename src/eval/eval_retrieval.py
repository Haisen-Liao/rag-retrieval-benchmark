import os
import json
import argparse
from collections import defaultdict


def load_qrels(qrels_path: str, min_rel: int = 1):
    """
    qrels.jsonl lines: {"qid": str/int, "doc_id": str, "relevance": int}
    We treat docs with relevance >= min_rel as relevant.
    Returns:
      rel: dict[qid] -> set(doc_id)
    """
    rel = defaultdict(set)

    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj["qid"])
            doc_id = str(obj["doc_id"])
            relevance = int(obj.get("relevance", 0))

            if relevance >= min_rel:
                rel[qid].add(doc_id)

    return rel


def load_run(run_path: str):
    """
    run.jsonl lines: {"qid": ..., "results": [{"doc_id":..., "score":...}, ...]}
    Returns:
      run: dict[qid] -> list(doc_id) (ranked)
    """
    run = {}

    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj["qid"])
            results = obj.get("results", [])

            # Keep ranking order
            doc_ids = [str(r["doc_id"]) for r in results if "doc_id" in r]
            run[qid] = doc_ids

    return run


def recall_at_k(rel_set, ranked_list, k: int) -> float:
    """
    Recall@K = (# relevant in top K) / (total relevant)
    If total relevant = 0, return None (we skip these queries in aggregation).
    """
    if not rel_set:
        return None
    topk = ranked_list[:k]
    hit = sum(1 for d in topk if d in rel_set)
    return hit / float(len(rel_set))


def mrr_at_k(rel_set, ranked_list, k: int) -> float:
    """
    MRR@K = 1 / rank of first relevant in top K (1-indexed), else 0.
    If total relevant = 0, return None (we skip these queries in aggregation).
    """
    if not rel_set:
        return None
    for i, d in enumerate(ranked_list[:k], start=1):
        if d in rel_set:
            return 1.0 / i
    return 0.0


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True, help="Path to qrels.jsonl")
    ap.add_argument("--run", required=True, help="Path to run.jsonl")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 5, 10, 100], help="List of K for Recall@K")
    ap.add_argument("--mrr_k", type=int, default=10, help="K for MRR@K")
    ap.add_argument(
        "--min_rel",
        type=int,
        default=1,
        help="Minimum relevance to be treated as relevant (default=1). For TREC-COVID, try 2.",
    )
    ap.add_argument("--out", required=False, default=None, help="Output metrics json path")
    args = ap.parse_args()

    qrels = load_qrels(args.qrels, min_rel=args.min_rel)
    run = load_run(args.run)

    ks = args.k
    mrr_k = args.mrr_k
    
    qrels_qids = set(qrels.keys())
    run_qids = set(run.keys())

    recall_lists = {k: [] for k in ks}
    mrr_list = []

    for qid in qrels_qids:
        rel_set = qrels.get(qid, set())
        ranked = run.get(qid, [])

        for k in ks:
            r = recall_at_k(rel_set, ranked, k)
            if r is not None:
                recall_lists[k].append(r)

        m = mrr_at_k(rel_set, ranked, mrr_k)
        if m is not None:
            mrr_list.append(m)

    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = mean(recall_lists[k])
    metrics[f"MRR@{mrr_k}"] = mean(mrr_list)

    metrics["min_rel"] = args.min_rel
    metrics["num_qrels_queries"] = len(qrels_qids)
    metrics["num_run_queries"] = len(run_qids)

    print("Metrics:")
    for k in ks:
        print(f"  Recall@{k}: {metrics[f'Recall@{k}']}")
    print(f"  MRR@{mrr_k}: {metrics[f'MRR@{mrr_k}']}")
    print(f"  min_rel: {metrics['min_rel']}")
    print(f"  num_qrels_queries: {metrics['num_qrels_queries']}")
    print(f"  num_run_queries: {metrics['num_run_queries']}")

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to: {args.out}")


if __name__ == "__main__":
    main()
