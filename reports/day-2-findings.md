# Day 2 Findings

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Product count: 12
- Evaluation query count: 20
- Embedding dimensions: 384

## Strategy comparison

| Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | Index ms | Mean query ms | P95 query ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| name | 87.50% | 100.00% | 100.00% | 100.00% | 90.00% | 35.00% | 21.00% | 10.50% | 507.418 | 8.163 | 12.776 |
| full | 97.50% | 100.00% | 100.00% | 100.00% | 100.00% | 35.00% | 21.00% | 10.50% | 45.048 | 8.802 | 11.715 |
| name_and_description | 97.50% | 100.00% | 100.00% | 100.00% | 100.00% | 35.00% | 21.00% | 10.50% | 136.845 | 6.648 | 7.484 |

## Concrete observations

- Best Recall@5: `name` with 100.00%.
- Best Precision@3: `name` with 35.00%.
- Name-only search has less descriptive information and may perform poorly for use-case queries.
- Adding descriptions usually helps synonym and use-case queries, but noisy descriptions may reduce precision.
- Category and brand can improve category or brand intent, but repeated catalog boilerplate may dominate embeddings.

## Intermediate representation inspected

### name

- SKU: `SHOE-RUN-001`
- Searchable text: `Velocity Men's Running Shoes`
- Vector shape: `[384]`
- First 10 vector values: `[-0.012216, 0.034925, -0.024714, -0.003388, 0.001437, -0.03096, -0.076519, -0.055064, -0.073804, -0.057849]`

### full

- SKU: `SHOE-RUN-001`
- Searchable text: `Name: Velocity Men's Running Shoes. Category: Men > Shoes > Running. Brand: Velocity. Description: Lightweight men's running shoes with breathable mesh and responsive cushioning for daily road running..`
- Vector shape: `[384]`
- First 10 vector values: `[-0.02048, 0.056252, -0.010915, 0.031883, -0.014058, -0.026249, -0.022565, -0.0548, -0.045468, -0.085569]`

### name_and_description

- SKU: `SHOE-RUN-001`
- Searchable text: `Name: Velocity Men's Running Shoes. Description: Lightweight men's running shoes with breathable mesh and responsive cushioning for daily road running..`
- Vector shape: `[384]`
- First 10 vector values: `[-0.035536, 0.054836, -0.010609, 0.055719, -0.002279, -0.032992, -0.013532, -0.041309, -0.05989, -0.051889]`

## Why recall matters before reranking

A reranker can only reorder candidates returned by the retrieval stage. If the relevant product is absent from Top K, the reranker cannot recover it.

For example, if the correct SKU is ranked 18th but the reranker receives only the first 10 candidates, the maximum possible final relevance for that query is already limited.

Therefore, first-stage retrieval should normally prioritize high Recall@K. The reranker can then improve the ordering and final Precision@K.

## Reproducibility

- Run: `python evaluate.py`
- Run tests: `pytest -q`
- Input catalog: `data/products.json`
- Golden dataset: `data/evaluation.json`
