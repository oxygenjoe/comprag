"""Entry point for ``python -m cumrag``."""

print(
    """\
CUMRAG - Comparative Utilization Metrics for Retrieval-Augmented Generation

Available modules:
  python -m cumrag.runner      Run evaluation harness
  python -m cumrag.evaluator   Score results with RAGAS/RAGChecker
  python -m cumrag.aggregator  Aggregate results with bootstrap CIs
  python -m cumrag.retriever   Query the vector index
  python -m cumrag.generator   Manage llama.cpp server

Scripts:
  python scripts/run_pipeline.py    End-to-end pipeline
  python scripts/visualize_results.py  Generate charts"""
)
