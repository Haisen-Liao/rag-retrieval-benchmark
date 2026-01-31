# RAG Retrieval Benchmark on BEIR

This project benchmarks retrieval methods for Retrieval-Augmented Generation (RAG) using BEIR datasets.

## Datasets
- **SciFact** (scientific claim verification)
- **TREC-COVID** (biomedical literature retrieval, graded relevance)

## Retrieval Methods
- BM25 (lexical retrieval)
- Dense retrieval (bi-encoder + FAISS)
- Hybrid retrieval (BM25 + Dense, alpha sweep)
- Hybrid + Fusion Reranking (cross-encoder, lambda sweep)

## Models
- Dense retriever: `BAAI/bge-small-en-v1.5`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Evaluation Metrics
- Recall@K
- MRR@10
- Graded relevance handled via configurable `min_rel` threshold (important for TREC-COVID)

## Key Findings
- Optimal hybrid weight (alpha) varies across datasets
- Fusion reranking improves early ranking quality (MRR@10, Recall@10)
- On datasets with many relevant documents per query (e.g., TREC-COVID), Recall@K values are inherently low and should be interpreted together with MRR/nDCG

## Reproducibility
Scripts are provided to:
1. Download and preprocess BEIR datasets
2. Build BM25 and FAISS indexes
3. Run retrieval and reranking
4. Evaluate results with configurable relevance thresholds