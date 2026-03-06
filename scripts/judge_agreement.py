#!/usr/bin/env python3
"""Judge agreement: compare frontier vs local fallback judge.

Score stratified samples with both Claude Sonnet 4.5 and Command R 35B Q4_K_M.
Compute Cohen's kappa overall and per stratum. If kappa >= 0.80, local fallback
is validated for reproducibility use.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure comprag package is importable when running as script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comprag.score import score_ragchecker  # noqa: E402

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "eval.yaml"
_MODELS_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"


def load_config() -> dict[str, Any]:
    """Load judge_validation config from eval.yaml."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["judge_validation"]


def load_model_roles() -> dict[str, str]:
    """Load model name -> role mapping from models.yaml."""
    with open(_MODELS_PATH) as f:
        cfg = yaml.safe_load(f)
    return {name: info["role"] for name, info in cfg["models"].items()}


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from a file or directory of .jsonl files."""
    records: list[dict[str, Any]] = []
    if path.is_dir():
        for f in sorted(path.glob("*.jsonl")):
            records.extend(_read_jsonl(f))
    else:
        records.extend(_read_jsonl(path))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a single JSONL file into a list of dicts."""
    items: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def stratified_sample(
    records: list[dict[str, Any]],
    model_roles: dict[str, str],
    primary_n: int,
    secondary_n: int,
) -> list[dict[str, Any]]:
    """Sample records stratified by model architecture role.

    Returns primary_n from primary-role models and secondary_n from
    secondary-role models. Falls back to available count if fewer exist.
    """
    primary = [r for r in records if model_roles.get(r["model"]) == "primary"]
    secondary = [r for r in records if model_roles.get(r["model"]) == "secondary"]

    random.seed(42)
    p_sample = random.sample(primary, min(primary_n, len(primary)))
    s_sample = random.sample(secondary, min(secondary_n, len(secondary)))

    logger.info(
        "Sampled %d primary + %d secondary records",
        len(p_sample), len(s_sample),
    )
    return p_sample + s_sample


def score_with_judge(
    record: dict[str, Any],
    judge_mode: str,
    judge_url: str = "http://localhost:8081",
) -> dict[str, float]:
    """Score a single record with the specified judge."""
    return score_ragchecker(
        query=record["query"],
        response=record["response"],
        context=record.get("context_chunks") or [],
        ground_truth=record["ground_truth"],
        judge_mode=judge_mode,
        judge_url=judge_url,
    )


def _discretize(value: float, thresholds: tuple[float, ...] = (0.33, 0.66)) -> int:
    """Bin a continuous [0,1] score into ordinal categories for kappa."""
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)


def cohens_kappa(
    labels_a: list[int],
    labels_b: list[int],
) -> float:
    """Compute Cohen's kappa between two raters.

    Uses the standard formula: kappa = (p_o - p_e) / (1 - p_e).
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    all_labels = sorted(set(labels_a) | set(labels_b))
    observed_agree = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    expected_agree = 0.0
    for label in all_labels:
        freq_a = labels_a.count(label) / n
        freq_b = labels_b.count(label) / n
        expected_agree += freq_a * freq_b

    if expected_agree >= 1.0:
        return 1.0
    return (observed_agree - expected_agree) / (1.0 - expected_agree)


def compute_agreement(
    frontier_scores: list[dict[str, float]],
    local_scores: list[dict[str, float]],
    strata: list[str],
    metric: str = "context_utilization",
) -> dict[str, Any]:
    """Compute Cohen's kappa overall and per stratum."""
    f_labels = [_discretize(s[metric]) for s in frontier_scores]
    l_labels = [_discretize(s[metric]) for s in local_scores]

    overall_kappa = cohens_kappa(f_labels, l_labels)

    per_stratum: dict[str, float] = {}
    for stratum in sorted(set(strata)):
        idx = [i for i, s in enumerate(strata) if s == stratum]
        f_sub = [f_labels[i] for i in idx]
        l_sub = [l_labels[i] for i in idx]
        per_stratum[stratum] = cohens_kappa(f_sub, l_sub)

    return {
        "overall_kappa": round(overall_kappa, 4),
        "per_stratum_kappa": {k: round(v, 4) for k, v in per_stratum.items()},
    }


def _score_all_samples(
    sampled: list[dict[str, Any]],
    model_roles: dict[str, str],
    judge_url: str,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[str]]:
    """Score every sample with both frontier and local judges.

    Returns (frontier_scores, local_scores, strata) parallel lists.
    """
    frontier_scores: list[dict[str, float]] = []
    local_scores: list[dict[str, float]] = []
    strata: list[str] = []

    for i, record in enumerate(sampled):
        role = model_roles.get(record["model"], "unknown")
        logger.info(
            "Scoring sample %d/%d (model=%s, role=%s)",
            i + 1, len(sampled), record["model"], role,
        )
        frontier_scores.append(score_with_judge(record, "frontier"))
        local_scores.append(score_with_judge(record, "local", judge_url))
        strata.append(role)

    return frontier_scores, local_scores, strata


def _build_result(
    agreement: dict[str, Any],
    strata: list[str],
    threshold: float,
) -> dict[str, Any]:
    """Assemble the final result dict with validation status."""
    validated = agreement["overall_kappa"] >= threshold
    return {
        "sample_size": len(strata),
        "primary_arch_n": strata.count("primary"),
        "secondary_arch_n": strata.count("secondary"),
        "agreement_threshold_kappa": threshold,
        "validated": validated,
        **agreement,
        "frontier_judge": "claude-sonnet-4-5-20250929",
        "local_judge": "c4ai-command-r-v01-Q4_K_M",
    }


def run_agreement(
    input_path: Path,
    sample_size: int,
    output_path: Path,
    judge_url: str = "http://localhost:8081",
) -> dict[str, Any]:
    """Run the full judge agreement pipeline.

    Loads records, samples, scores with both judges, computes kappa.
    """
    config = load_config()
    model_roles = load_model_roles()

    primary_n = config.get("primary_arch_sample", sample_size // 2)
    secondary_n = config.get("secondary_arch_sample", sample_size // 2)

    records = load_records(input_path)
    sampled = stratified_sample(records, model_roles, primary_n, secondary_n)

    frontier_scores, local_scores, strata = _score_all_samples(
        sampled, model_roles, judge_url,
    )
    agreement = compute_agreement(frontier_scores, local_scores, strata)
    threshold = config.get("agreement_threshold_kappa", 0.80)
    result = _build_result(agreement, strata, threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote agreement results to %s", output_path)
    logger.info(
        "Overall kappa=%.4f, validated=%s (threshold=%.2f)",
        agreement["overall_kappa"], result["validated"], threshold,
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Judge agreement validation: frontier vs local fallback.",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to scored JSONL file or directory of JSONL files.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=100,
        help="Total sample size (default: 100).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("judge_agreement.json"),
        help="Output path for agreement results JSON (default: judge_agreement.json).",
    )
    parser.add_argument(
        "--judge-url", type=str, default="http://localhost:8081",
        help="Base URL for local judge server (default: http://localhost:8081).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    run_agreement(
        input_path=args.input,
        sample_size=args.sample_size,
        output_path=args.output,
        judge_url=args.judge_url,
    )


if __name__ == "__main__":
    main()
