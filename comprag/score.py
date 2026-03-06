"""Evaluation scoring: thin wrapper around RAGChecker with judge routing.

Supports two judge modes:
- "frontier": Anthropic Claude Sonnet 4.5 via litellm (used by RAGChecker)
- "local": Command R 35B Q4_K_M on llama.cpp server (port 8081)

The generation server (8080) must be stopped before starting the local judge.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from ragchecker import RAGChecker, RAGResult, RAGResults
from ragchecker.container import RetrievedDoc

logger = logging.getLogger(__name__)

# --- Constants from config/eval.yaml ---
_DEFAULT_EVAL_PATH = Path(__file__).resolve().parent.parent / "config" / "eval.yaml"

# Judge constants (locked).
FRONTIER_JUDGE_MODEL = "claude-sonnet-4-5-20250929"
FRONTIER_JUDGE_PROVIDER = "anthropic"
LOCAL_JUDGE_MODEL = "c4ai-command-r-v01"
LOCAL_JUDGE_PORT = 8081
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024

# RAGChecker metrics we care about for comprag.
TARGET_METRICS = ["context_utilization", "self_knowledge", "noise_sensitivity_in_relevant"]


def _load_eval_config(path: Path = _DEFAULT_EVAL_PATH) -> dict[str, Any]:
    """Load eval config from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def _build_checker_frontier() -> RAGChecker:
    """Build RAGChecker using Anthropic Claude Sonnet 4.5 as judge.

    Requires ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY for frontier judge")

    model_name = f"anthropic/{FRONTIER_JUDGE_MODEL}"
    logger.info("Initializing frontier judge: %s", model_name)

    return RAGChecker(
        extractor_name=model_name,
        checker_name=model_name,
        batch_size_extractor=8,
        batch_size_checker=8,
    )


def _build_checker_local(judge_url: str) -> RAGChecker:
    """Build RAGChecker using local llama.cpp Command R server as judge.

    The local server must expose an OpenAI-compatible /v1/chat/completions
    endpoint at judge_url.
    """
    # For local models via OpenAI-compat API, use openai/ prefix with api_base.
    model_name = f"openai/{LOCAL_JUDGE_MODEL}"
    logger.info("Initializing local judge: %s at %s", model_name, judge_url)

    return RAGChecker(
        extractor_name=model_name,
        checker_name=model_name,
        extractor_api_base=f"{judge_url}/v1",
        checker_api_base=f"{judge_url}/v1",
        batch_size_extractor=4,
        batch_size_checker=4,
    )


def _build_checker(judge_mode: str, judge_url: str) -> RAGChecker:
    """Route to frontier or local judge based on mode."""
    if judge_mode == "frontier":
        return _build_checker_frontier()
    if judge_mode == "local":
        return _build_checker_local(judge_url)
    raise ValueError(f"Unknown judge_mode '{judge_mode}'. Expected 'frontier' or 'local'.")


def _build_rag_result(
    query: str,
    response: str,
    context: list[str],
    ground_truth: str,
) -> RAGResult:
    """Construct a single RAGResult from raw inputs."""
    retrieved_docs = [
        RetrievedDoc(doc_id=str(i), text=chunk)
        for i, chunk in enumerate(context)
    ]
    return RAGResult(
        query_id="q0",
        query=query,
        gt_answer=ground_truth,
        response=response,
        retrieved_context=retrieved_docs,
    )


def _extract_target_metrics(result: RAGResult) -> dict[str, float]:
    """Pull target metric values from the evaluated RAGResult."""
    return {
        "context_utilization": result.metrics.get("context_utilization", 0.0),
        "self_knowledge": result.metrics.get("self_knowledge", 0.0),
        "noise_sensitivity": result.metrics.get("noise_sensitivity_in_relevant", 0.0),
    }


def score_ragchecker(
    query: str,
    response: str,
    context: list[str],
    ground_truth: str,
    judge_mode: str = "frontier",
    judge_url: str = "http://localhost:8081",
) -> dict[str, float]:
    """Score a single RAG result using RAGChecker.

    Returns:
        {"context_utilization": float, "self_knowledge": float,
         "noise_sensitivity": float}

    Args:
        query: The user's question.
        response: The model's generated response.
        context: Retrieved text chunks provided to the model.
        ground_truth: Reference answer for evaluation.
        judge_mode: "frontier" (Anthropic API) or "local" (Command R on llama.cpp).
        judge_url: Base URL of local judge server (only used when judge_mode="local").
    """
    checker = _build_checker(judge_mode, judge_url)
    rag_result = _build_rag_result(query, response, context, ground_truth)
    results = RAGResults(results=[rag_result])

    logger.info("Running RAGChecker evaluation (judge_mode=%s)", judge_mode)
    checker.evaluate(results, metrics=TARGET_METRICS)

    metrics = _extract_target_metrics(results.results[0])
    logger.info("RAGChecker scores: %s", metrics)
    return metrics
