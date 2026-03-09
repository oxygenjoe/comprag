#!/usr/bin/env python3
"""Sonnet 4.6 validation: re-score 500 Command R records, compute kappa agreement.

Samples 250 from primary architectures, 250 from secondary (per eval.yaml).
Outputs scored records + kappa summary.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comprag.score import score_ragchecker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SCORED_DIR = Path(__file__).resolve().parent.parent / "results" / "scored"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "sonnet_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SONNET_MODEL = "claude-sonnet-4-6"

# Primary arch files (scored by Command R)
PRIMARY_FILES = [
    "qwen2.5-14b-instruct_Q4_K_M_rgb_pass2_loose.jsonl",
    "qwen2.5-14b-instruct_Q8_0_rgb_pass2_loose.jsonl",
    "llama-3.1-8b-instruct_Q4_K_M_rgb_pass2_loose.jsonl",
    "llama-3.1-8b-instruct_Q8_0_rgb_pass2_loose.jsonl",
]

# Secondary arch files + pass variety (scored by Command R)
SECONDARY_FILES = [
    "gemma-2-9b-instruct_Q4_K_M_rgb_pass2_loose.jsonl",
    "gemma-2-9b-instruct_Q8_0_rgb_pass2_loose.jsonl",
    "gemma-2-9b-instruct_Q4_K_M_rgb_pass1_baseline.jsonl",
    "gemma-2-9b-instruct_Q4_K_M_rgb_pass3_strict.jsonl",
    "mistral-nemo-12b-instruct_Q4_K_M_rgb_pass2_loose.jsonl",
]

METRICS = [
    "overall_precision", "overall_recall", "overall_f1",
    "context_utilization", "faithfulness", "hallucination",
    "self_knowledge", "noise_sensitivity_relevant",
]

# 3-bin discretization for kappa (same as DeepSeek validation)
BINS = [0.0, 0.33, 0.67, 1.01]
BIN_LABELS = [0, 1, 2]


def load_records(filenames: list[str], n: int, seed: int = 42) -> list[dict]:
    """Load and sample n records from multiple scored files."""
    all_recs = []
    for fname in filenames:
        path = SCORED_DIR / fname
        if not path.exists():
            logger.warning("Missing %s", path)
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rec["_source_file"] = fname
                    all_recs.append(rec)

    random.seed(seed)
    random.shuffle(all_recs)
    return all_recs[:n]


def discretize(val: float) -> int:
    for i in range(len(BINS) - 1):
        if val < BINS[i + 1]:
            return BIN_LABELS[i]
    return BIN_LABELS[-1]


def cohens_kappa_linear(y1: list[int], y2: list[int], n_classes: int = 3) -> float:
    """Linear weighted Cohen's kappa."""
    n = len(y1)
    if n == 0:
        return 0.0

    # Confusion matrix
    mat = np.zeros((n_classes, n_classes), dtype=float)
    for a, b in zip(y1, y2):
        mat[a][b] += 1.0

    # Weight matrix (linear)
    weights = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i][j] = abs(i - j) / (n_classes - 1)

    # Expected
    row_sums = mat.sum(axis=1)
    col_sums = mat.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n

    observed_disagreement = (weights * mat).sum() / n
    expected_disagreement = (weights * expected).sum() / n

    if expected_disagreement == 0:
        return 1.0
    return 1.0 - observed_disagreement / expected_disagreement


def main():
    logger.info("=== Sonnet 4.6 Validation Pass ===")

    # Sample records
    primary = load_records(PRIMARY_FILES, 250)
    secondary = load_records(SECONDARY_FILES, 250)
    sample = primary + secondary
    logger.info("Sampled %d primary + %d secondary = %d records",
                len(primary), len(secondary), len(sample))

    output_path = OUTPUT_DIR / "sonnet_validation_scored.jsonl"
    kappa_path = OUTPUT_DIR / "sonnet_validation_kappa.json"

    # Resume support
    done_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["query_id"])

    remaining = [r for r in sample if r.get("query_id", "") not in done_ids]
    logger.info("%d already done, %d remaining", len(done_ids), len(remaining))

    # Score with Sonnet
    with open(output_path, "a") as out:
        for i, rec in enumerate(remaining):
            query = rec.get("query", "")
            response = rec.get("response", "")
            context = rec.get("context_chunks") or []
            gt = rec.get("ground_truth", "")

            try:
                sonnet_scores = score_ragchecker(
                    query, response, context, gt,
                    judge_provider="anthropic",
                    judge_model=SONNET_MODEL,
                )
            except Exception as e:
                logger.error("FAIL record %d: %s", i, e)
                sonnet_scores = {k: 0.0 for k in METRICS}

            result = {
                "query_id": rec.get("query_id", ""),
                "model": rec.get("model", ""),
                "quantization": rec.get("quantization", ""),
                "pass": rec.get("pass", ""),
                "subset": rec.get("subset", ""),
                "source_file": rec.get("_source_file", ""),
                "commandr_scores": rec.get("scores", {}).get("ragchecker", {}),
                "sonnet_scores": sonnet_scores,
            }
            out.write(json.dumps(result) + "\n")
            out.flush()

            if (i + 1) % 5 == 0 or i + 1 == len(remaining):
                logger.info("[%d/%d] scored", i + 1, len(remaining))

    # Compute kappa
    logger.info("Computing agreement...")
    records = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    kappa_results = {}
    for metric in METRICS:
        cr_vals = []
        sn_vals = []
        raw_agree = 0
        for r in records:
            cr = r["commandr_scores"].get(metric, 0.0)
            sn = r["sonnet_scores"].get(metric, 0.0)
            cr_bin = discretize(cr)
            sn_bin = discretize(sn)
            cr_vals.append(cr_bin)
            sn_vals.append(sn_bin)
            if cr_bin == sn_bin:
                raw_agree += 1

        k = cohens_kappa_linear(cr_vals, sn_vals)
        kappa_results[metric] = {
            "kappa_linear": round(k, 3),
            "raw_agreement": f"{raw_agree}/{len(records)}",
            "n": len(records),
        }
        logger.info("%-30s κ=%.3f  agree=%d/%d", metric, k, raw_agree, len(records))

    with open(kappa_path, "w") as f:
        json.dump(kappa_results, f, indent=2)

    logger.info("Results: %s", output_path)
    logger.info("Kappa: %s", kappa_path)
    logger.info("=== Validation complete ===")


if __name__ == "__main__":
    main()
