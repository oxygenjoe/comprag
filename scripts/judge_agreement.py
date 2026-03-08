#!/usr/bin/env python3
"""Judge agreement: frontier-vs-frontier validation across 3 API judges.

Score stratified samples with Claude Opus 4.6, GPT-5.4, and Gemini 3 Flash.
Compute pairwise Cohen's kappa for all 3 pairs. If a cheaper judge hits
kappa >= 0.80 against Opus, document it as a validated cost-efficient alternative.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from itertools import combinations
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
    """Sample records stratified by model architecture role."""
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
    judge_provider: str,
    judge_model: str,
) -> dict[str, float]:
    """Score a single record with the specified frontier judge."""
    return score_ragchecker(
        query=record["query"],
        response=record["response"],
        context=record.get("context_chunks") or [],
        ground_truth=record["ground_truth"],
        judge_provider=judge_provider,
        judge_model=judge_model,
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


def _score_all_samples(
    sampled: list[dict[str, Any]],
    judges: list[dict[str, str]],
    model_roles: dict[str, str],
) -> tuple[dict[str, list[dict[str, float]]], list[str]]:
    """Score every sample with all frontier judges.

    Returns (judge_scores, strata) where judge_scores maps
    "provider/model" to a parallel list of score dicts.
    """
    judge_scores: dict[str, list[dict[str, float]]] = {}
    for judge in judges:
        key = f"{judge['provider']}/{judge['model_id']}"
        judge_scores[key] = []

    strata: list[str] = []

    for i, record in enumerate(sampled):
        role = model_roles.get(record["model"], "unknown")
        strata.append(role)
        for judge in judges:
            key = f"{judge['provider']}/{judge['model_id']}"
            logger.info(
                "Scoring sample %d/%d with %s (model=%s, role=%s)",
                i + 1, len(sampled), key, record["model"], role,
            )
            scores = score_with_judge(record, judge["provider"], judge["model_id"])
            judge_scores[key].append(scores)

    return judge_scores, strata


def compute_pairwise_agreement(
    judge_scores: dict[str, list[dict[str, float]]],
    strata: list[str],
    metric: str = "context_utilization",
) -> list[dict[str, Any]]:
    """Compute pairwise Cohen's kappa for all judge pairs."""
    judge_keys = sorted(judge_scores.keys())
    results: list[dict[str, Any]] = []

    for key_a, key_b in combinations(judge_keys, 2):
        a_labels = [_discretize(s[metric]) for s in judge_scores[key_a]]
        b_labels = [_discretize(s[metric]) for s in judge_scores[key_b]]

        overall = cohens_kappa(a_labels, b_labels)

        per_stratum: dict[str, float] = {}
        for stratum in sorted(set(strata)):
            idx = [i for i, s in enumerate(strata) if s == stratum]
            a_sub = [a_labels[i] for i in idx]
            b_sub = [b_labels[i] for i in idx]
            per_stratum[stratum] = round(cohens_kappa(a_sub, b_sub), 4)

        results.append({
            "judge_a": key_a,
            "judge_b": key_b,
            "overall_kappa": round(overall, 4),
            "per_stratum_kappa": per_stratum,
        })

    return results


def run_agreement(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the full frontier-vs-frontier judge agreement pipeline."""
    config = load_config()
    model_roles = load_model_roles()

    judges = config.get("judges", [])
    if len(judges) < 2:
        raise RuntimeError("Need at least 2 judges in eval.yaml judge_validation.judges")

    primary_n = config.get("primary_arch_sample", 50)
    secondary_n = config.get("secondary_arch_sample", 50)
    threshold = config.get("agreement_threshold_kappa", 0.80)

    records = load_records(input_path)
    sampled = stratified_sample(records, model_roles, primary_n, secondary_n)

    judge_scores, strata = _score_all_samples(sampled, judges, model_roles)
    pairwise = compute_pairwise_agreement(judge_scores, strata)

    # Check which cheaper judges validate against primary (first in list).
    primary_key = f"{judges[0]['provider']}/{judges[0]['model_id']}"
    validated_alternatives: list[str] = []
    for pair in pairwise:
        if primary_key in (pair["judge_a"], pair["judge_b"]):
            other = pair["judge_b"] if pair["judge_a"] == primary_key else pair["judge_a"]
            if pair["overall_kappa"] >= threshold:
                validated_alternatives.append(other)

    result = {
        "sample_size": len(sampled),
        "primary_arch_n": strata.count("primary"),
        "secondary_arch_n": strata.count("secondary"),
        "agreement_threshold_kappa": threshold,
        "primary_judge": primary_key,
        "pairwise_results": pairwise,
        "validated_alternatives": validated_alternatives,
        "judges": [f"{j['provider']}/{j['model_id']}" for j in judges],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote agreement results to %s", output_path)

    for pair in pairwise:
        logger.info(
            "Kappa %s vs %s: %.4f",
            pair["judge_a"], pair["judge_b"], pair["overall_kappa"],
        )
    if validated_alternatives:
        logger.info("Validated alternatives (kappa >= %.2f): %s", threshold, validated_alternatives)
    else:
        logger.info("No alternatives met kappa >= %.2f threshold", threshold)

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Judge agreement: frontier-vs-frontier pairwise kappa.",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to scored JSONL file or directory of JSONL files.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("judge_agreement.json"),
        help="Output path for agreement results JSON (default: judge_agreement.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    run_agreement(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
