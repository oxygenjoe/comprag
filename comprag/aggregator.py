"""Results aggregator with bootstrap confidence intervals.

Reads raw JSONL eval results, groups by (model x quantization x hardware_tier
x dataset x eval_subset), computes 95% bootstrap CIs (1000 resamples), and
outputs summary tables in CSV, Markdown, and JSONL formats.

Supports both v1 (flat top-level fields) and v2 (nested run_config/perf/metrics)
raw JSONL input schemas.  Output defaults to the v2 aggregated schema.

Importable as module; also runnable as CLI via:
    python -m comprag.aggregator --input results/raw/ --output results/aggregated/

Flags any result where CI width > 15% of the mean for additional runs.
Warns if fewer than 3 runs exist for any combination.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy import stats as scipy_stats

from comprag.utils import get_hardware_meta, get_logger, read_jsonl

log = get_logger("comprag.aggregator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROUP_KEYS = ("model", "quantization", "hardware_tier", "dataset", "eval_subset", "pass")

QUALITY_METRICS = (
    "faithfulness",
    "context_utilization",
    "self_knowledge",
    "noise_sensitivity",
    "answer_relevancy",
    "negative_rejection_rate",
)

PERFORMANCE_METRICS = (
    "tokens_per_second",
    "ttft_ms",
    "vram_usage_mb",
    "gpu_temp",
)

ALL_METRICS = QUALITY_METRICS + PERFORMANCE_METRICS

# v2 field-name aliases: v2_name -> canonical metric name.
# The runner's perf block uses shortened names; we map them to our canonical
# metric names so both old and new records contribute to the same aggregated
# metric.
_PERF_ALIASES: dict[str, str] = {
    "tokens_per_sec": "tokens_per_second",
    "time_to_first_token_ms": "ttft_ms",
    "vram_mb": "vram_usage_mb",
}

# Reverse map: canonical -> set of aliases (used during extraction)
_CANONICAL_TO_ALIASES: dict[str, list[str]] = {}
for _alias, _canon in _PERF_ALIASES.items():
    _CANONICAL_TO_ALIASES.setdefault(_canon, []).append(_alias)

# v2 group-key mapping: run_config field name -> canonical GROUP_KEYS name
_GROUP_KEY_MAP: dict[str, str] = {
    "model": "model",
    "quant": "quantization",
    "hardware": "hardware_tier",
}

MIN_RUNS = 3
BOOTSTRAP_RESAMPLES = 1000
CI_LEVEL = 0.95
CI_WIDTH_THRESHOLD = 0.15  # Flag if CI width > 15% of mean


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(input_path: Union[str, Path]) -> list[dict]:
    """Load all JSONL files from a directory or a single file.

    Args:
        input_path: Path to a directory containing .jsonl files,
                    or path to a single .jsonl file.

    Returns:
        List of parsed result records.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If no JSONL files found in directory.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        records = list(read_jsonl(input_path))
        log.info("Loaded %d records from %s", len(records), input_path)
        return records

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    jsonl_files = sorted(input_path.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {input_path}")

    records: list[dict] = []
    for fp in jsonl_files:
        file_records = list(read_jsonl(fp))
        log.info("Loaded %d records from %s", len(file_records), fp.name)
        records.extend(file_records)

    log.info("Total records loaded: %d from %d files", len(records), len(jsonl_files))
    return records


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def _extract_group_key(record: dict) -> tuple:
    """Extract the grouping key tuple from a result record.

    Handles both v1 (flat top-level) and v2 (nested run_config) formats:
      - v2: run_config.model, run_config.quant -> quantization,
            run_config.hardware -> hardware_tier
      - v2: subset -> eval_subset
      - v1 fallback: top-level model, quantization, hardware_tier, etc.

    Returns:
        Tuple of (model, quantization, hardware_tier, dataset, eval_subset).
        Missing fields default to 'unknown'.
    """
    run_config = record.get("run_config", {}) or {}

    values: list[str] = []
    for gk in GROUP_KEYS:
        # 1. Check run_config using the reverse map
        found = False
        for rc_field, canon in _GROUP_KEY_MAP.items():
            if canon == gk and rc_field in run_config:
                val = run_config[rc_field]
                values.append(str(val) if val else "unknown")
                found = True
                break

        if found:
            continue

        # 2. v2 uses "subset" for eval_subset
        if gk == "eval_subset" and "subset" in record:
            val = record["subset"]
            values.append(str(val) if val else "unknown")
            continue

        # 3. "pass" field: check run_config.pass, then top-level, default to pass2_loose
        if gk == "pass":
            val = run_config.get("pass", record.get("pass", "pass2_loose"))
            values.append(str(val) if val else "pass2_loose")
            continue

        # 4. Top-level fallback (v1 format)
        val = record.get(gk, "unknown")
        values.append(str(val) if val else "unknown")

    return tuple(values)


def _try_float(val: object) -> Optional[float]:
    """Attempt to convert a value to float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _extract_metric(record: dict, metric_name: str) -> Optional[float]:
    """Extract a metric value from a record.

    Search order (first non-None wins):
      1. perf dict (v2 performance block)
      2. eval_metrics dict
      3. eval_spec_metrics dict
      4. metrics dict (v2 quality metrics / v1 nested metrics)
      5. top-level field (v1 flat format)

    Within each dict, the canonical metric_name is tried first, then any
    known aliases (e.g. ``tokens_per_sec`` for ``tokens_per_second``).

    Returns:
        Float value or None if not present or not numeric.
    """
    # Build the list of names to search: canonical + aliases
    names_to_try = [metric_name] + _CANONICAL_TO_ALIASES.get(metric_name, [])

    # Ordered list of nested dicts to probe
    nested_dicts = ("perf", "eval_metrics", "eval_spec_metrics", "metrics")

    for dict_key in nested_dicts:
        nested = record.get(dict_key)
        if not isinstance(nested, dict):
            continue
        for name in names_to_try:
            if name in nested:
                result = _try_float(nested[name])
                if result is not None:
                    return result

    # Top-level fallback (v1 flat format)
    for name in names_to_try:
        if name in record:
            result = _try_float(record[name])
            if result is not None:
                return result

    return None


def group_results(records: list[dict]) -> dict[tuple, list[dict]]:
    """Group result records by the canonical key tuple.

    Args:
        records: List of parsed JSONL result dicts.

    Returns:
        Dict mapping (model, quant, hw, dataset, subset) -> list of records.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = _extract_group_key(record)
        groups[key].append(record)
    return dict(groups)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence_level: float = CI_LEVEL,
    rng_seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for the mean of values.

    Uses scipy.stats.bootstrap (BCa method) with the specified number of
    resamples. Falls back to simple percentile method if BCa fails.

    Args:
        values: 1D array of metric values.
        n_resamples: Number of bootstrap resamples (default 1000).
        confidence_level: Confidence level (default 0.95).
        rng_seed: Random seed for reproducibility.

    Returns:
        Dict with keys: mean, ci_low, ci_high, ci_width, n, flagged.
        'flagged' is True if CI width > 15% of mean.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean_val = float(np.mean(values))

    if n < 2:
        # Cannot compute CI with fewer than 2 values
        return {
            "mean": mean_val,
            "ci_low": mean_val,
            "ci_high": mean_val,
            "ci_width": 0.0,
            "n": n,
            "flagged": False,
        }

    rng = np.random.default_rng(rng_seed)

    try:
        result = scipy_stats.bootstrap(
            (values,),
            statistic=np.mean,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=rng,
            method="BCa",
        )
        ci_low = float(result.confidence_interval.low)
        ci_high = float(result.confidence_interval.high)
    except Exception as e:
        # Fallback: percentile bootstrap
        log.warning("BCa bootstrap failed (%s), falling back to percentile method", e)
        boot_means = np.empty(n_resamples)
        for i in range(n_resamples):
            sample = rng.choice(values, size=n, replace=True)
            boot_means[i] = np.mean(sample)

        alpha = 1.0 - confidence_level
        ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    ci_width = ci_high - ci_low

    # Flag if CI width > 15% of mean (avoid division by zero)
    flagged = False
    if abs(mean_val) > 1e-10:
        flagged = ci_width > CI_WIDTH_THRESHOLD * abs(mean_val)

    return {
        "mean": mean_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_width,
        "n": n,
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_group(records: list[dict]) -> dict:
    """Compute aggregated statistics for a single group of records.

    For each metric, collects all non-None values and computes bootstrap CI.

    Args:
        records: List of records in this group.

    Returns:
        Dict with metric names as keys, each mapping to a bootstrap_ci result.
        Also includes 'n_runs' and 'warnings' list.
    """
    result: dict = {"n_runs": len(records), "warnings": [], "metrics": {}}

    if len(records) < MIN_RUNS:
        result["warnings"].append(
            f"Only {len(records)} run(s) — minimum {MIN_RUNS} required for reliable CIs"
        )

    for metric_name in ALL_METRICS:
        values = []
        for rec in records:
            val = _extract_metric(rec, metric_name)
            if val is not None:
                values.append(val)

        if not values:
            result["metrics"][metric_name] = None
            continue

        ci_result = bootstrap_ci(np.array(values))
        result["metrics"][metric_name] = ci_result

        if ci_result["flagged"]:
            result["warnings"].append(
                f"{metric_name}: CI width ({ci_result['ci_width']:.4f}) "
                f"> 15% of mean ({ci_result['mean']:.4f}) — needs more runs"
            )

    return result


def aggregate_all(records: list[dict]) -> dict[tuple, dict]:
    """Aggregate all records, grouped by canonical key.

    Args:
        records: All parsed result records.

    Returns:
        Dict mapping group key tuple -> aggregated stats dict.
    """
    groups = group_results(records)
    aggregated: dict[tuple, dict] = {}

    for key, group_records in sorted(groups.items()):
        key_label = " | ".join(f"{k}={v}" for k, v in zip(GROUP_KEYS, key))
        log.info("Aggregating group: %s (%d runs)", key_label, len(group_records))
        aggregated[key] = aggregate_group(group_records)

    return aggregated


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------


def _build_flat_rows(aggregated: dict[tuple, dict]) -> list[dict]:
    """Flatten aggregated results into rows suitable for CSV/Markdown.

    Each row has the group key fields, n_runs, and for each metric:
    {metric}_mean, {metric}_ci_low, {metric}_ci_high, {metric}_flagged.
    """
    rows: list[dict] = []

    for key, agg in sorted(aggregated.items()):
        row: dict = {}
        for gk, gv in zip(GROUP_KEYS, key):
            row[gk] = gv
        row["n_runs"] = agg["n_runs"]

        for metric_name in ALL_METRICS:
            ci = agg["metrics"].get(metric_name)
            if ci is None:
                row[f"{metric_name}_mean"] = ""
                row[f"{metric_name}_ci_low"] = ""
                row[f"{metric_name}_ci_high"] = ""
                row[f"{metric_name}_flagged"] = ""
            else:
                row[f"{metric_name}_mean"] = f"{ci['mean']:.4f}"
                row[f"{metric_name}_ci_low"] = f"{ci['ci_low']:.4f}"
                row[f"{metric_name}_ci_high"] = f"{ci['ci_high']:.4f}"
                row[f"{metric_name}_flagged"] = "YES" if ci["flagged"] else ""

        has_warnings = bool(agg.get("warnings"))
        row["warnings"] = "; ".join(agg.get("warnings", [])) if has_warnings else ""
        rows.append(row)

    return rows


def _csv_fieldnames() -> list[str]:
    """Generate ordered CSV column names."""
    fields = list(GROUP_KEYS) + ["n_runs"]
    for metric_name in ALL_METRICS:
        fields.extend([
            f"{metric_name}_mean",
            f"{metric_name}_ci_low",
            f"{metric_name}_ci_high",
            f"{metric_name}_flagged",
        ])
    fields.append("warnings")
    return fields


def write_csv(aggregated: dict[tuple, dict], output_path: Union[str, Path]) -> Path:
    """Write aggregated results to a CSV file.

    Args:
        aggregated: Output from aggregate_all().
        output_path: Path to write CSV file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _build_flat_rows(aggregated)
    fieldnames = _csv_fieldnames()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("CSV written: %s (%d rows)", output_path, len(rows))
    return output_path


# ---------------------------------------------------------------------------
# Output: Markdown
# ---------------------------------------------------------------------------


def format_markdown(aggregated: dict[tuple, dict]) -> str:
    """Format aggregated results as a Markdown table.

    Shows a compact summary: group keys, n_runs, and for each quality metric
    the mean with CI range. Performance metrics are in a separate section.

    Args:
        aggregated: Output from aggregate_all().

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    # Quality metrics table
    lines.append("## Quality Metrics (95% CI)")
    lines.append("")

    q_headers = list(GROUP_KEYS) + ["n"] + [
        m for m in QUALITY_METRICS
    ]
    lines.append("| " + " | ".join(q_headers) + " |")
    lines.append("| " + " | ".join("---" for _ in q_headers) + " |")

    for key, agg in sorted(aggregated.items()):
        cells = list(key) + [str(agg["n_runs"])]
        for metric_name in QUALITY_METRICS:
            ci = agg["metrics"].get(metric_name)
            if ci is None:
                cells.append("N/A")
            else:
                flag = " **!**" if ci["flagged"] else ""
                cells.append(
                    f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]{flag}"
                )
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # Performance metrics table
    lines.append("## Performance Metrics (95% CI)")
    lines.append("")

    p_headers = list(GROUP_KEYS) + ["n"] + [
        m for m in PERFORMANCE_METRICS
    ]
    lines.append("| " + " | ".join(p_headers) + " |")
    lines.append("| " + " | ".join("---" for _ in p_headers) + " |")

    for key, agg in sorted(aggregated.items()):
        cells = list(key) + [str(agg["n_runs"])]
        for metric_name in PERFORMANCE_METRICS:
            ci = agg["metrics"].get(metric_name)
            if ci is None:
                cells.append("N/A")
            else:
                flag = " **!**" if ci["flagged"] else ""
                cells.append(
                    f"{ci['mean']:.1f} [{ci['ci_low']:.1f}, {ci['ci_high']:.1f}]{flag}"
                )
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # Warnings section
    all_warnings: list[str] = []
    for key, agg in sorted(aggregated.items()):
        if agg.get("warnings"):
            key_label = " | ".join(f"{k}={v}" for k, v in zip(GROUP_KEYS, key))
            for w in agg["warnings"]:
                all_warnings.append(f"- **{key_label}**: {w}")

    if all_warnings:
        lines.append("## Warnings")
        lines.append("")
        lines.extend(all_warnings)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output: JSONL summary
# ---------------------------------------------------------------------------


def write_jsonl_summary(
    aggregated: dict[tuple, dict],
    output_path: Union[str, Path],
) -> Path:
    """Write aggregated results as v2 JSONL, one record per group.

    Each record matches the v2 aggregated output schema::

        {
          "timestamp": "2026-03-01T14:30:00Z",
          "model": "...",
          "quantization": "...",
          "hardware_tier": "...",
          "dataset": "...",
          "eval_subset": "...",
          "n_runs": 5,
          "metrics": { "<name>": <mean_float>, ... },
          "metrics_ci": { "<name>": { "mean", "ci_low", "ci_high", ... }, ... },
          "hardware_meta": { "gpu", "driver", "framework", "os" },
          "warnings": [...]
        }

    The ``metrics`` block contains bare mean values for quick consumption;
    ``metrics_ci`` holds the full bootstrap CI detail.

    Args:
        aggregated: Output from aggregate_all().
        output_path: Path to write JSONL file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hw_meta = get_hardware_meta()

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for key, agg in sorted(aggregated.items()):
            key_dict = dict(zip(GROUP_KEYS, key))

            # v2 flat metrics block: metric_name -> mean value
            flat_metrics: dict[str, object] = {}
            # v2 detailed CI block
            ci_metrics: dict[str, dict] = {}

            for metric_name in ALL_METRICS:
                ci = agg["metrics"].get(metric_name)
                if ci is not None:
                    flat_metrics[metric_name] = round(ci["mean"], 6)
                    ci_metrics[metric_name] = {
                        "mean": round(ci["mean"], 6),
                        "ci_low": round(ci["ci_low"], 6),
                        "ci_high": round(ci["ci_high"], 6),
                        "ci_width": round(ci["ci_width"], 6),
                        "n": ci["n"],
                        "flagged": ci["flagged"],
                    }

            record: dict = {
                "timestamp": timestamp,
                "model": key_dict["model"],
                "quantization": key_dict["quantization"],
                "hardware_tier": key_dict["hardware_tier"],
                "dataset": key_dict["dataset"],
                "eval_subset": key_dict["eval_subset"],
                "n_runs": agg["n_runs"],
                "metrics": flat_metrics,
                "metrics_ci": ci_metrics,
                "hardware_meta": hw_meta,
                "warnings": agg.get("warnings", []),
            }

            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            f.write(line + "\n")
            count += 1

    log.info("JSONL summary written: %s (%d records)", output_path, count)
    return output_path


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def run_aggregation(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    formats: Optional[list[str]] = None,
    print_markdown: bool = True,
) -> dict:
    """Run the full aggregation pipeline.

    Args:
        input_path: Path to directory of JSONL files or a single JSONL file.
        output_dir: Directory to write aggregated output files.
        formats: List of output formats to produce. Any of 'csv', 'markdown',
                 'json'. Defaults to all three.
        print_markdown: If True, print markdown table to stdout.

    Returns:
        Dict with 'aggregated' data, 'output_files' list, and 'warnings' count.
    """
    if formats is None:
        formats = ["csv", "markdown", "json"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and aggregate
    records = load_results(input_path)
    if not records:
        log.warning("No records found in %s", input_path)
        return {"aggregated": {}, "output_files": [], "warnings": 0}

    aggregated = aggregate_all(records)

    output_files: list[str] = []
    total_warnings = sum(len(a.get("warnings", [])) for a in aggregated.values())

    # Write outputs
    if "csv" in formats:
        csv_path = write_csv(aggregated, output_dir / "summary.csv")
        output_files.append(str(csv_path))

    if "json" in formats:
        jsonl_path = write_jsonl_summary(aggregated, output_dir / "summary.jsonl")
        output_files.append(str(jsonl_path))

    if "markdown" in formats:
        md_text = format_markdown(aggregated)
        md_path = output_dir / "summary.md"
        md_path.write_text(md_text, encoding="utf-8")
        output_files.append(str(md_path))
        log.info("Markdown written: %s", md_path)

        if print_markdown:
            print(md_text)

    # Summary
    n_groups = len(aggregated)
    n_flagged = sum(
        1 for a in aggregated.values()
        if any(
            m is not None and m.get("flagged", False)
            for m in a["metrics"].values()
        )
    )
    log.info(
        "Aggregation complete: %d groups, %d flagged, %d warnings",
        n_groups,
        n_flagged,
        total_warnings,
    )

    return {
        "aggregated": aggregated,
        "output_files": output_files,
        "warnings": total_warnings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the aggregator."""
    parser = argparse.ArgumentParser(
        description=(
            "CUMRAG Results Aggregator — compute bootstrap CIs from raw JSONL results"
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/raw/",
        help="Path to raw JSONL results directory or file (default: results/raw/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/aggregated/",
        help="Output directory for aggregated results (default: results/aggregated/)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "markdown", "json"],
        default=None,
        help="Output format (default: all formats). Can be specified multiple times.",
        action="append",
        dest="formats",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress markdown output to stdout",
    )

    args = parser.parse_args()

    formats = args.formats if args.formats else ["csv", "markdown", "json"]

    try:
        result = run_aggregation(
            input_path=args.input,
            output_dir=args.output,
            formats=formats,
            print_markdown=not args.quiet,
        )
    except FileNotFoundError as e:
        log.error("Input not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        log.error("Invalid input: %s", e)
        sys.exit(1)

    if result["warnings"] > 0:
        log.warning(
            "%d warning(s) — some groups may need additional runs",
            result["warnings"],
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
