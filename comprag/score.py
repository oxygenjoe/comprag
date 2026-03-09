"""Evaluation scoring: thin wrapper around RAGChecker with local judge.

Judge model: Command R 35B Q4_K_M (local, llama-server on port 8081).
The judge_provider and judge_model parameters allow switching to Anthropic
API for the judge agreement validation check.
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

# Default judge (locked).
DEFAULT_JUDGE_PROVIDER = "local"
DEFAULT_JUDGE_MODEL = "c4ai-command-r-v01"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024

# RAGChecker metrics — capture all 11.
TARGET_METRICS = "all_metrics"

# Provider -> litellm model prefix and API key env var.
_JUDGE_PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "anthropic": {"prefix": "anthropic", "env": "ANTHROPIC_API_KEY"},
    "local": {"prefix": "openai", "env": "_LOCAL_JUDGE",
              "base_url": "http://localhost:8080/v1"},
}


def _load_eval_config(path: Path = _DEFAULT_EVAL_PATH) -> dict[str, Any]:
    """Load eval config from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def _build_checker(
    judge_provider: str = DEFAULT_JUDGE_PROVIDER,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> RAGChecker:
    """Build RAGChecker using a judge model.

    Args:
        judge_provider: One of 'local', 'anthropic'.
        judge_model: Model ID for the judge (e.g. 'c4ai-command-r-v01').
    """
    cfg = _JUDGE_PROVIDER_CONFIG.get(judge_provider)
    if not cfg:
        raise ValueError(
            f"Unknown judge_provider '{judge_provider}'. "
            f"Expected one of: {list(_JUDGE_PROVIDER_CONFIG.keys())}"
        )

    if judge_provider == "local":
        # Local llama-server: set litellm env vars for OpenAI-compat endpoint
        os.environ["OPENAI_API_KEY"] = "not-needed"
        os.environ["OPENAI_API_BASE"] = cfg["base_url"]
        model_name = f"openai/{judge_model}"
    else:
        api_key = os.environ.get(cfg["env"])
        if not api_key:
            raise RuntimeError(f"Missing {cfg['env']} for {judge_provider} judge")
        if "base_url" in cfg:
            os.environ["OPENAI_API_BASE"] = cfg["base_url"]
        model_name = f"{cfg['prefix']}/{judge_model}"

    logger.info("Initializing judge: %s", model_name)

    return RAGChecker(
        extractor_name=model_name,
        checker_name=model_name,
        batch_size_extractor=8,
        batch_size_checker=8,
    )


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


# RAGChecker internal name -> spec-compliant raw record name.
_RAGCHECKER_TO_SPEC: dict[str, str] = {
    "precision": "overall_precision",
    "recall": "overall_recall",
    "f1": "overall_f1",
    "claim_recall": "claim_recall",
    "context_precision": "context_precision",
    "context_utilization": "context_utilization",
    "noise_sensitivity_in_relevant": "noise_sensitivity_relevant",
    "noise_sensitivity_in_irrelevant": "noise_sensitivity_irrelevant",
    "hallucination": "hallucination",
    "self_knowledge": "self_knowledge",
    "faithfulness": "faithfulness",
}


def _extract_target_metrics(result: RAGResult) -> dict[str, float]:
    """Pull all RAGChecker metric values, mapped to spec field names."""
    return {
        spec_name: result.metrics.get(rc_name, 0.0)
        for rc_name, spec_name in _RAGCHECKER_TO_SPEC.items()
    }


def score_ragchecker(
    query: str,
    response: str,
    context: list[str],
    ground_truth: str,
    judge_provider: str = DEFAULT_JUDGE_PROVIDER,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, float]:
    """Score a single RAG result using RAGChecker.

    Returns:
        Dict with all 11 RAGChecker metrics mapped to spec field names:
        overall_precision, overall_recall, overall_f1, claim_recall,
        context_precision, context_utilization, self_knowledge,
        noise_sensitivity_relevant, noise_sensitivity_irrelevant,
        hallucination, faithfulness.

    Args:
        judge_provider: 'local' | 'anthropic'. Selects judge backend.
        judge_model: Model ID for the judge.
    """
    checker = _build_checker(judge_provider, judge_model)
    rag_result = _build_rag_result(query, response, context, ground_truth)
    results = RAGResults(results=[rag_result])

    logger.info("Running RAGChecker evaluation (judge=%s/%s)", judge_provider, judge_model)
    checker.evaluate(results, metrics=TARGET_METRICS)

    metrics = _extract_target_metrics(results.results[0])
    logger.info("RAGChecker scores: %s", metrics)
    return metrics
