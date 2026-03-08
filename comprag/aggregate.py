"""Bootstrap statistics and Preference_Gap computation for CompRAG.

Reads scored JSONL, groups by (model, quantization, dataset, subset, pass),
bootstraps all RAGChecker metrics + Preference_Gap, flags capability-degraded configs.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CAPABILITY_DEGRADATION_THRESHOLD = 0.30
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

# Raw record RAGChecker field name -> aggregated short name (spec contract).
_RAW_TO_AGG: dict[str, str] = {
    "overall_precision": "overall_precision",
    "overall_recall": "overall_recall",
    "overall_f1": "overall_f1",
    "claim_recall": "claim_recall",
    "context_precision": "context_precision",
    "context_utilization": "cu",
    "self_knowledge": "sk",
    "noise_sensitivity_relevant": "ns_relevant",
    "noise_sensitivity_irrelevant": "ns_irrelevant",
    "hallucination": "hallucination",
    "faithfulness": "faithfulness",
}


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    """
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty array")

    rng = np.random.default_rng(42)
    mean = float(np.mean(values))
    n = len(values)

    resample_means = np.array([
        float(np.mean(rng.choice(values, size=n, replace=True)))
        for _ in range(n_resamples)
    ])

    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(resample_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(resample_means, 100 * (1 - alpha / 2)))

    return (mean, ci_lo, ci_hi)


def compute_preference_gap(
    pass2_records: list[dict], pass3_records: list[dict]
) -> dict[str, float]:
    """Per-query: pass3_cu - pass2_cu. Then bootstrap over queries.

    Returns {"mean": ..., "ci_lo": ..., "ci_hi": ..., "std": ...}.
    """
    pass2_by_qid = {
        r["query_id"]: r["scores"]["ragchecker"]["context_utilization"]
        for r in pass2_records
    }
    pass3_by_qid = {
        r["query_id"]: r["scores"]["ragchecker"]["context_utilization"]
        for r in pass3_records
    }

    common_qids = sorted(set(pass2_by_qid) & set(pass3_by_qid))
    if not common_qids:
        logger.warning("No overlapping query_ids between pass2 and pass3")
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "std": 0.0}

    diffs = np.array([
        pass3_by_qid[qid] - pass2_by_qid[qid] for qid in common_qids
    ])

    mean, ci_lo, ci_hi = bootstrap_ci(diffs)
    std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    return {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "std": std}


def _bootstrap_metric(values: list[float]) -> dict[str, float]:
    """Bootstrap a single metric array, returning mean/ci_lo/ci_hi/std."""
    arr = np.array(values)
    mean, ci_lo, ci_hi = bootstrap_ci(arr)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "std": std}


def _group_key(record: dict) -> tuple[str, str, str, str, str]:
    """Extract group key from a scored record."""
    return (
        record["model"],
        record["quantization"],
        record["dataset"],
        record["subset"],
        record["pass"],
    )


def aggregate_results(scored_dir: str, output_dir: str | None = None) -> list[dict]:
    """Group by (model, quant, dataset, subset, pass).

    Bootstrap CU/SK/NS/Preference_Gap.
    Flag capability-degraded configs (pass3_cu below threshold).
    Write aggregated JSONL.
    """
    scored_path = Path(scored_dir)
    records: list[dict] = []

    for jsonl_file in sorted(scored_path.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    logger.info("Loaded %d scored records from %s", len(records), scored_dir)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        groups[_group_key(rec)] = groups.get(_group_key(rec), [])
        groups[_group_key(rec)].append(rec)

    # Index pass2 records by (model, quant, dataset, subset) for pref gap
    pass2_index: dict[tuple, list[dict]] = defaultdict(list)
    for key, recs in groups.items():
        model, quant, dataset, subset, pass_name = key
        if pass_name.startswith("pass2"):
            pass2_index[(model, quant, dataset, subset)] = recs

    results: list[dict] = []
    for key in sorted(groups.keys()):
        model, quant, dataset, subset, pass_name = key
        group_records = groups[key]

        metrics = {}
        for raw_key, agg_key in _RAW_TO_AGG.items():
            vals = [r["scores"]["ragchecker"].get(raw_key, 0.0)
                    for r in group_records]
            metrics[agg_key] = _bootstrap_metric(vals)

        # Preference gap: only for pass3, needs matching pass2
        base_key = (model, quant, dataset, subset)
        if pass_name.startswith("pass3") and base_key in pass2_index:
            metrics["preference_gap"] = compute_preference_gap(
                pass2_index[base_key], group_records
            )
        else:
            metrics["preference_gap"] = {
                "mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "std": 0.0,
            }

        # Capability degradation: pass3 CU mean below threshold
        is_degraded = (
            pass_name.startswith("pass3")
            and metrics["cu"]["mean"] < CAPABILITY_DEGRADATION_THRESHOLD
        )

        source = group_records[0].get("source", "local")

        result = {
            "model": model,
            "quantization": quant,
            "source": source,
            "dataset": dataset,
            "subset": subset,
            "pass": pass_name,
            "n_queries": len(group_records),
            "metrics": metrics,
            "capability_degraded": is_degraded,
        }
        results.append(result)

    # Write aggregated JSONL
    out = Path(output_dir) if output_dir else scored_path.parent / "aggregated"
    out.mkdir(parents=True, exist_ok=True)
    output_file = out / "aggregated.jsonl"

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info("Wrote %d aggregated records to %s", len(results), output_file)
    return results
