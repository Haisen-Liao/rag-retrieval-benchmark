import json

path = r"runs/retrieval/scifact_hybrid_a08_rerank_top100.jsonl"
with open(path, "r", encoding="utf-8") as f:
    obj = json.loads(next(f))

print("First qid:", obj["qid"])
print("Num results:", len(obj["results"]))
print("Top3:", obj["results"][:3])
