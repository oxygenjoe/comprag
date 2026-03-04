"""Entry point for ``python -m comprag``."""

print(
    """\
CompRAG - Comparative Retrieval-Augmented Generation Evaluation

Available modules:
  python -m comprag.runner      Run evaluation harness
  python -m comprag.evaluator   Score results with RAGAS/RAGChecker
  python -m comprag.aggregator  Aggregate results with bootstrap CIs
  python -m comprag.retriever   Query the vector index
  python -m comprag.generator   Manage llama.cpp server

Scripts:
  python scripts/run_pipeline.py    End-to-end pipeline
  python scripts/visualize_results.py  Generate charts"""
)
