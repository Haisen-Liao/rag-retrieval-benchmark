# RAG Retrieval Benchmark on BEIR

This repository benchmarks retrieval methods for Retrieval-Augmented Generation (RAG) on BEIR datasets, with a focus on lexical, dense, hybrid retrieval and reranking strategies.

## Datasets
- **SciFact**: Scientific claim verification dataset
- **TREC-COVID**: Biomedical literature retrieval dataset with graded relevance labels

## Retrieval Methods
- **BM25** (lexical retrieval)
- **Dense retrieval** (bi-encoder + FAISS)
- **Hybrid retrieval** (BM25 + Dense, alpha sweep)
- **Hybrid + Fusion Reranking** (cross-encoder reranker, lambda sweep)

## Models
- Dense retriever: `BAAI/bge-small-en-v1.5`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Evaluation Metrics
- Recall@K
- MRR@10
- Graded relevance handled via configurable `min_rel` threshold  
  (important for datasets such as TREC-COVID)

## Key Findings
- Optimal hybrid weight (alpha) varies across datasets
- Fusion reranking consistently improves early ranking quality (MRR@10, Recall@10)
- For datasets with many relevant documents per query (e.g., TREC-COVID), Recall@K values are inherently low and should be interpreted together with MRR/nDCG-style metrics

---

## Quickstart

### 1. Environment setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt