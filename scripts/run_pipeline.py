#!/usr/bin/env python3
"""End-to-end CompRAG evaluation pipeline orchestrator.

Chains all pipeline stages in order:
    download -> normalize -> validate_datasets -> build_index
    -> validate_index -> run (dry-run mock) -> aggregate -> visualize

Each step: logs start/end with timing, checks if output exists
(skip unless --force), handles failures gracefully, returns exit code.

Flags:
    --dry-run   Mock LLM, generate synthetic results matching v2 JSONL schema
    --steps     Run only specific steps (comma-separated or repeated)
    --force     Re-run completed steps even if output exists

Dry-run generates synthetic raw JSONL results for testing the
aggregation and visualization stages without a llama.cpp server.

Standalone:
    python scripts/run_pipeline.py --dry-run
    python scripts/run_pipeline.py --steps download,normalize --force
    python scripts/run_pipeline.py --dry-run --steps aggregate,visualize

Importable:
    from scripts.run_pipeline import Pipeline, run_pipeline
"""

import argparse
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Project root detection and path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from comprag.utils import Timer, get_hardware_meta, get_logger, setup_logging

logger = get_logger("comprag.pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_STEPS = [
    "download",
    "normalize",
    "validate_datasets",
    "build_index",
    "validate_index",
    "run",
    "aggregate",
    "visualize",
]

# Datasets and subsets for synthetic result generation
SYNTH_DATASETS = {
    "rgb": ["noise_robustness", "negative_rejection"],
    "nq": ["test"],
    "halueval": ["qa"],
}

SYNTH_MODELS = [
    ("llama-3.1-8b-instruct", "Q4_K_M"),
    ("qwen2.5-7b-instruct", "Q4_K_M"),
    ("smollm2-1.7b-instruct", "Q8_0"),
]

SYNTH_HARDWARE = "v100"
SYNTH_SAMPLES_PER_COMBO = 10
SYNTH_SEEDS = [42, 43, 44]

# Output paths (relative to project root)
RAW_RESULTS_DIR = _PROJECT_ROOT / "results" / "raw"
AGGREGATED_DIR = _PROJECT_ROOT / "results" / "aggregated"
FIGURES_DIR = _PROJECT_ROOT / "results" / "figures"
DATASETS_DIR = _PROJECT_ROOT / "datasets"
INDEX_DIR = _PROJECT_ROOT / "index"


# ---------------------------------------------------------------------------
# Step result tracking
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of a single pipeline step."""

    name: str
    status: str = "pending"  # pending, skipped, success, error
    elapsed_sec: float = 0.0
    message: str = ""
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Aggregated result of the full pipeline run."""

    steps: list[StepResult] = field(default_factory=list)
    dry_run: bool = False

    @property
    def succeeded(self) -> int:
        return sum(1 for s in self.steps if s.status == "success")

    @property
    def failed(self) -> int:
        return sum(1 for s in self.steps if s.status == "error")

    @property
    def skipped(self) -> int:
        return sum(1 for s in self.steps if s.status == "skipped")

    @property
    def exit_code(self) -> int:
        """0 if no errors, 1 if any step failed."""
        return 1 if self.failed > 0 else 0


# ---------------------------------------------------------------------------
# Synthetic result generation (dry-run mode)
# ---------------------------------------------------------------------------


def generate_synthetic_results(
    output_dir: Path,
    seed: int = 42,
) -> Path:
    """Generate synthetic raw JSONL results matching the v2 schema.

    Produces plausible evaluation results for testing the aggregation
    and visualization pipeline without requiring a running llama.cpp server.

    Args:
        output_dir: Directory to write the synthetic JSONL file.
        seed: Random seed for reproducibility.

    Returns:
        Path to the generated JSONL file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dry_run_synthetic.jsonl"

    rng = random.Random(seed)
    hw_meta = get_hardware_meta()
    records_written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for model, quant in SYNTH_MODELS:
            for dataset, subsets in SYNTH_DATASETS.items():
                for subset in subsets:
                    for run_seed in SYNTH_SEEDS:
                        for sample_idx in range(SYNTH_SAMPLES_PER_COMBO):
                            record = _make_synthetic_record(
                                rng=rng,
                                model=model,
                                quant=quant,
                                dataset=dataset,
                                subset=subset,
                                sample_idx=sample_idx,
                                run_seed=run_seed,
                                hw_meta=hw_meta,
                            )
                            line = json.dumps(
                                record, ensure_ascii=False, separators=(",", ":")
                            )
                            f.write(line + "\n")
                            records_written += 1

    logger.info(
        "Synthetic results: %d records written to %s", records_written, output_path
    )
    return output_path


def _make_synthetic_record(
    rng: random.Random,
    model: str,
    quant: str,
    dataset: str,
    subset: str,
    sample_idx: int,
    run_seed: int,
    hw_meta: dict,
) -> dict:
    """Build a single synthetic v2 JSONL result record.

    All required fields from the task spec:
        sample_id, dataset, subset, query, ground_truth, response,
        retrieved_chunks, run_config (model, quant, hardware, seed),
        perf (ttft_ms, total_tokens, tokens_per_sec, wall_clock_sec,
        vram_mb, gpu_temp), metrics, timestamp, error
    """
    sample_id = f"{dataset}_{subset}_{sample_idx:04d}"

    # Plausible quality scores scaled by model "size"
    model_quality = {
        "llama-3.1-8b-instruct": 0.72,
        "qwen2.5-7b-instruct": 0.68,
        "smollm2-1.7b-instruct": 0.45,
    }
    base_q = model_quality.get(model, 0.6)

    # Add per-seed jitter for bootstrap CI testing
    jitter = rng.gauss(0, 0.03)

    faithfulness = max(0.05, min(0.98, base_q + jitter + rng.gauss(0, 0.02)))
    context_util = max(0.05, min(0.95, base_q * 0.9 + rng.gauss(0, 0.02)))
    self_knowledge = max(0.05, min(0.95, base_q * 0.85 + rng.gauss(0, 0.02)))
    noise_sensitivity = max(0.05, min(0.6, 0.3 + rng.gauss(0, 0.03)))
    answer_relevancy = max(0.05, min(0.98, base_q * 0.95 + rng.gauss(0, 0.02)))
    neg_rejection = max(0.05, min(0.95, base_q * 0.8 + rng.gauss(0, 0.03)))

    # Performance metrics
    model_tps = {
        "llama-3.1-8b-instruct": 32.0,
        "qwen2.5-7b-instruct": 35.0,
        "smollm2-1.7b-instruct": 55.0,
    }
    base_tps = model_tps.get(model, 30.0)
    tokens_per_sec = max(1.0, base_tps + rng.gauss(0, 2.0))
    total_tokens = rng.randint(80, 350)
    wall_clock = total_tokens / tokens_per_sec
    ttft = max(10.0, 50 + rng.gauss(0, 10))

    model_vram = {
        "llama-3.1-8b-instruct": 6300,
        "qwen2.5-7b-instruct": 6000,
        "smollm2-1.7b-instruct": 2500,
    }
    vram_mb = model_vram.get(model, 5000) + rng.randint(-100, 100)
    gpu_temp = 65 + rng.randint(-5, 10)

    query_text = f"[DRY-RUN] Synthetic query {sample_idx} for {dataset}/{subset}"
    ground_truth = f"[DRY-RUN] Expected answer for sample {sample_idx}"
    response_text = f"[DRY-RUN] Model response for sample {sample_idx} (seed={run_seed})"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "subset": subset,
        "query": query_text,
        "ground_truth": ground_truth,
        "response": response_text,
        "retrieved_chunks": [
            {
                "chunk_id": f"chunk_{dataset}_{i}",
                "text": f"[DRY-RUN] Retrieved chunk {i} for {query_text[:40]}",
                "score": round(rng.uniform(0.5, 0.95), 4),
            }
            for i in range(5)
        ],
        "run_config": {
            "model": model,
            "quant": quant,
            "hardware": SYNTH_HARDWARE,
            "seed": run_seed,
        },
        "perf": {
            "ttft_ms": round(ttft, 2),
            "total_tokens": total_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "wall_clock_sec": round(wall_clock, 3),
            "vram_mb": vram_mb,
            "gpu_temp": gpu_temp,
        },
        "metrics": {
            "faithfulness": round(faithfulness, 4),
            "context_utilization": round(context_util, 4),
            "self_knowledge": round(self_knowledge, 4),
            "noise_sensitivity": round(noise_sensitivity, 4),
            "answer_relevancy": round(answer_relevancy, 4),
            "negative_rejection_rate": round(neg_rejection, 4),
        },
        "timestamp": timestamp,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Pipeline step implementations
# ---------------------------------------------------------------------------


def _step_download(force: bool, dry_run: bool) -> StepResult:
    """Download evaluation datasets."""
    result = StepResult(name="download")

    if dry_run:
        result.status = "skipped"
        result.message = "Skipped in dry-run mode (not needed for synthetic results)"
        return result

    # Check if datasets already exist
    if not force and DATASETS_DIR.is_dir():
        has_content = any(DATASETS_DIR.iterdir())
        if has_content:
            result.status = "skipped"
            result.message = "Datasets directory exists; use --force to re-download"
            return result

    try:
        from scripts.download_datasets import download_all

        paths = download_all(datasets_dir=DATASETS_DIR, force=force)
        result.status = "success"
        result.message = f"Downloaded {len(paths)} dataset(s): {', '.join(paths.keys())}"
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.message = f"Download failed: {e}"
        logger.error("Download step failed: %s", e)

    return result


def _step_normalize(force: bool, dry_run: bool) -> StepResult:
    """Normalize datasets into unified JSONL schema."""
    result = StepResult(name="normalize")

    if dry_run:
        result.status = "skipped"
        result.message = "Skipped in dry-run mode"
        return result

    # Check if normalized files exist
    if not force:
        normalized_dirs = list(DATASETS_DIR.glob("*/normalized"))
        has_normalized = any(
            list(d.glob("*.jsonl")) for d in normalized_dirs if d.is_dir()
        )
        if has_normalized:
            result.status = "skipped"
            result.message = "Normalized data exists; use --force to re-normalize"
            return result

    try:
        from scripts.normalize_datasets import normalize_all

        results = normalize_all(datasets_dir=DATASETS_DIR)
        total_subsets = sum(len(v) for v in results.values())
        result.status = "success"
        result.message = (
            f"Normalized {len(results)} dataset(s), {total_subsets} subset(s)"
        )
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.message = f"Normalization failed: {e}"
        logger.error("Normalize step failed: %s", e)

    return result


def _step_validate_datasets(force: bool, dry_run: bool) -> StepResult:
    """Validate normalized datasets against unified schema."""
    result = StepResult(name="validate_datasets")

    if dry_run:
        result.status = "skipped"
        result.message = "Skipped in dry-run mode"
        return result

    try:
        from scripts.validate_datasets import validate_datasets

        vr = validate_datasets(datasets_dir=DATASETS_DIR)
        if vr.total_records == 0:
            result.status = "error"
            result.error = "No normalized records found to validate"
            result.message = "No records found — run normalize first"
        elif vr.is_valid:
            result.status = "success"
            result.message = (
                f"Validated {vr.total_records} records, "
                f"{vr.valid_records} valid, 0 violations"
            )
        else:
            result.status = "error"
            result.error = f"{vr.violation_count} schema violation(s) found"
            result.message = (
                f"Validated {vr.total_records} records, "
                f"{vr.violation_count} violation(s)"
            )
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.message = f"Dataset validation failed: {e}"
        logger.error("Validate datasets step failed: %s", e)

    return result


def _step_build_index(force: bool, dry_run: bool) -> StepResult:
    """Build ChromaDB vector index from dataset documents."""
    result = StepResult(name="build_index")

    if dry_run:
        result.status = "skipped"
        result.message = "Skipped in dry-run mode"
        return result

    # Check if index exists
    if not force and INDEX_DIR.is_dir():
        chroma_files = list(INDEX_DIR.glob("**/chroma*")) + list(
            INDEX_DIR.glob("**/*.bin")
        )
        if chroma_files:
            result.status = "skipped"
            result.message = "Index directory exists; use --force to rebuild"
            return result

    try:
        from scripts.build_index import build_all_indexes

        build_all_indexes(force=force)
        result.status = "success"
        result.message = "Index build complete"
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.message = f"Index build failed: {e}"
        logger.error("Build index step failed: %s", e)

    return result


def _step_validate_index(force: bool, dry_run: bool) -> StepResult:
    """Validate ChromaDB index integrity."""
    result = StepResult(name="validate_index")

    if dry_run:
        result.status = "skipped"
        result.message = "Skipped in dry-run mode"
        return result

    if not INDEX_DIR.is_dir():
        result.status = "error"
        result.error = "Index directory does not exist"
        result.message = "Index directory not found — run build_index first"
        return result

    try:
        from scripts.validate_index import audit_index

        reports = audit_index(index_dir=str(INDEX_DIR))
        if not reports:
            result.status = "error"
            result.error = "No collections found in index"
            result.message = "No collections to validate"
        else:
            passed = sum(1 for r in reports if r.passed)
            total = len(reports)
            if passed == total:
                result.status = "success"
                result.message = f"All {total} collection(s) passed validation"
            else:
                result.status = "error"
                result.error = f"{total - passed}/{total} collection(s) failed"
                result.message = (
                    f"Index validation: {passed}/{total} collections passed"
                )
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.message = f"Index validation failed: {e}"
        logger.error("Validate index step failed: %s", e)

    return result


def _step_run(force: bool, dry_run: bool) -> StepResult:
    """Run evaluation (dry-run generates synthetic results)."""
    result = StepResult(name="run")

    if dry_run:
        # Check if synthetic results already exist
        synth_file = RAW_RESULTS_DIR / "dry_run_synthetic.jsonl"
        if not force and synth_file.is_file() and synth_file.stat().st_size > 0:
            result.status = "skipped"
            result.message = (
                f"Synthetic results exist at {synth_file}; use --force to regenerate"
            )
            return result

        try:
            output_path = generate_synthetic_results(RAW_RESULTS_DIR)
            n_lines = sum(1 for _ in open(output_path, "r"))
            result.status = "success"
            result.message = (
                f"Generated {n_lines} synthetic results at {output_path}"
            )
        except Exception as e:
            result.status = "error"
            result.error = str(e)
            result.message = f"Synthetic result generation failed: {e}"
            logger.error("Run (dry-run) step failed: %s", e)
    else:
        # Real run requires llama.cpp server
        # Check if raw results already exist
        if not force and RAW_RESULTS_DIR.is_dir():
            existing = list(RAW_RESULTS_DIR.glob("*.jsonl"))
            if existing:
                result.status = "skipped"
                result.message = (
                    f"Raw results exist ({len(existing)} file(s)); "
                    f"use --force to re-run"
                )
                return result

        result.status = "error"
        result.error = "Real evaluation requires llama.cpp server (not implemented in pipeline yet)"
        result.message = (
            "Use --dry-run for synthetic results, or run "
            "comprag.runner directly for real evaluation"
        )

    return result


def _step_aggregate(force: bool, dry_run: bool) -> StepResult:
    """Aggregate raw results with bootstrap CIs."""
    result = StepResult(name="aggregate")

    # Check if raw results exist
    if not RAW_RESULTS_DIR.is_dir():
        result.status = "error"
        result.error = "No raw results directory found"
        result.message = "Raw results directory not found — run eval first"
        return result

    raw_files = list(RAW_RESULTS_DIR.glob("*.jsonl"))
    if not raw_files:
        result.status = "error"
        result.error = "No JSONL files in raw results directory"
        result.message = "No raw result files found — run eval first"
        return result

    # Check if aggregated results exist
    summary_csv = AGGREGATED_DIR / "summary.csv"
    if not force and summary_csv.is_file() and summary_csv.stat().st_size > 0:
        result.status = "skipped"
        result.message = (
            f"Aggregated results exist at {summary_csv}; use --force to re-aggregate"
        )
        return result

    try:
        from comprag.aggregator import run_aggregation

        agg_result = run_aggregation(
            input_path=str(RAW_RESULTS_DIR),
            output_dir=str(AGGREGATED_DIR),
            print_markdown=False,
        )
        n_groups = len(agg_result.get("aggregated", {}))
        n_files = len(agg_result.get("output_files", []))
        n_warnings = agg_result.get("warnings", 0)
        result.status = "success"
        result.message = (
            f"Aggregated {n_groups} group(s), wrote {n_files} file(s), "
            f"{n_warnings} warning(s)"
        )
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.message = f"Aggregation failed: {e}"
        logger.error("Aggregate step failed: %s", e)

    return result


def _step_visualize(force: bool, dry_run: bool) -> StepResult:
    """Generate publication-quality charts from aggregated data."""
    result = StepResult(name="visualize")

    summary_csv = AGGREGATED_DIR / "summary.csv"
    if not summary_csv.is_file():
        result.status = "error"
        result.error = "Aggregated summary.csv not found"
        result.message = "No aggregated data — run aggregate step first"
        return result

    # Check if figures exist
    if not force and FIGURES_DIR.is_dir():
        existing_pngs = list(FIGURES_DIR.glob("*.png"))
        if len(existing_pngs) >= 6:
            result.status = "skipped"
            result.message = (
                f"{len(existing_pngs)} chart(s) exist; use --force to regenerate"
            )
            return result

    try:
        from scripts.visualize_results import generate_all_charts, load_aggregated_csv

        df = load_aggregated_csv(summary_csv)
        charts = generate_all_charts(df, FIGURES_DIR)
        result.status = "success"
        result.message = f"Generated {len(charts)} chart(s) in {FIGURES_DIR}"
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.message = f"Visualization failed: {e}"
        logger.error("Visualize step failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEP_REGISTRY: dict[str, Callable[[bool, bool], StepResult]] = {
    "download": _step_download,
    "normalize": _step_normalize,
    "validate_datasets": _step_validate_datasets,
    "build_index": _step_build_index,
    "validate_index": _step_validate_index,
    "run": _step_run,
    "aggregate": _step_aggregate,
    "visualize": _step_visualize,
}


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class Pipeline:
    """End-to-end evaluation pipeline orchestrator.

    Args:
        steps: List of step names to execute (default: all steps).
        dry_run: If True, mock the LLM and generate synthetic results.
        force: If True, re-run steps even if output exists.
    """

    def __init__(
        self,
        steps: Optional[list[str]] = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> None:
        self.steps = steps or list(ALL_STEPS)
        self.dry_run = dry_run
        self.force = force

        # Validate step names
        invalid = set(self.steps) - set(ALL_STEPS)
        if invalid:
            raise ValueError(
                f"Unknown step(s): {invalid}. Valid steps: {ALL_STEPS}"
            )

    def run(self) -> PipelineResult:
        """Execute all configured pipeline steps in order.

        Returns:
            PipelineResult with per-step status and timing.
        """
        pipeline_result = PipelineResult(dry_run=self.dry_run)

        mode_label = "DRY-RUN" if self.dry_run else "LIVE"
        force_label = " (--force)" if self.force else ""
        logger.info(
            "Pipeline starting [%s%s] — %d step(s): %s",
            mode_label,
            force_label,
            len(self.steps),
            ", ".join(self.steps),
        )
        print(f"\n{'='*60}")
        print(f"CompRAG Pipeline [{mode_label}{force_label}]")
        print(f"Steps: {', '.join(self.steps)}")
        print(f"{'='*60}\n")

        pipeline_start = time.perf_counter()
        critical_failure = False

        for step_name in self.steps:
            if step_name not in STEP_REGISTRY:
                sr = StepResult(
                    name=step_name, status="error", error=f"Unknown step: {step_name}"
                )
                pipeline_result.steps.append(sr)
                continue

            step_fn = STEP_REGISTRY[step_name]
            print(f"  [{step_name}] Starting...")
            logger.info("Step '%s' starting", step_name)

            with Timer() as t:
                try:
                    sr = step_fn(self.force, self.dry_run)
                except Exception as e:
                    sr = StepResult(
                        name=step_name,
                        status="error",
                        error=f"Unhandled exception: {e}",
                        message=f"Step crashed: {e}",
                    )
                    logger.exception("Step '%s' crashed", step_name)

            sr.elapsed_sec = t.elapsed

            status_icon = {
                "success": "OK",
                "skipped": "SKIP",
                "error": "FAIL",
                "pending": "???",
            }.get(sr.status, sr.status)

            print(
                f"  [{step_name}] {status_icon} ({sr.elapsed_sec:.2f}s) — {sr.message}"
            )
            logger.info(
                "Step '%s' finished: %s (%.2fs) — %s",
                step_name,
                sr.status,
                sr.elapsed_sec,
                sr.message,
            )

            pipeline_result.steps.append(sr)

            # On critical failure in non-dry-run data steps, continue but log
            if sr.status == "error":
                # Aggregate and visualize depend on earlier steps' output;
                # if run/aggregate fail we can still try subsequent steps
                # since they check for their own prerequisites
                logger.warning(
                    "Step '%s' failed — continuing with remaining steps", step_name
                )

        pipeline_elapsed = time.perf_counter() - pipeline_start

        # Print summary
        print(f"\n{'='*60}")
        print("Pipeline Summary")
        print(f"{'='*60}")
        print(f"  Total time: {pipeline_elapsed:.2f}s")
        print(
            f"  Success: {pipeline_result.succeeded}  "
            f"Skipped: {pipeline_result.skipped}  "
            f"Failed: {pipeline_result.failed}"
        )

        for sr in pipeline_result.steps:
            status_icon = {
                "success": "OK  ",
                "skipped": "SKIP",
                "error": "FAIL",
            }.get(sr.status, "??? ")
            print(f"    [{status_icon}] {sr.name:.<25} {sr.elapsed_sec:.2f}s")

        exit_code = pipeline_result.exit_code
        print(f"\n  Exit code: {exit_code}")
        print(f"{'='*60}\n")

        return pipeline_result


def run_pipeline(
    steps: Optional[list[str]] = None,
    dry_run: bool = False,
    force: bool = False,
) -> PipelineResult:
    """Convenience function to run the pipeline.

    Args:
        steps: List of step names (default: all).
        dry_run: Mock LLM, generate synthetic results.
        force: Re-run completed steps.

    Returns:
        PipelineResult with per-step outcomes.
    """
    pipeline = Pipeline(steps=steps, dry_run=dry_run, force=force)
    return pipeline.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "CompRAG end-to-end evaluation pipeline orchestrator.\n\n"
            "Chains: download -> normalize -> validate_datasets -> build_index\n"
            "     -> validate_index -> run -> aggregate -> visualize"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --dry-run                    # Full pipeline with synthetic data\n"
            "  %(prog)s --dry-run --force             # Force re-run all steps\n"
            "  %(prog)s --steps run,aggregate,visualize --dry-run\n"
            "  %(prog)s --steps download,normalize    # Real data, specific steps\n"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock LLM, generate synthetic results matching v2 JSONL schema",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help=(
            "Comma-separated list of steps to run. "
            f"Valid: {', '.join(ALL_STEPS)}. Default: all steps."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run completed steps even if output exists",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Returns:
        Exit code: 0 on success, 1 on any failure.
    """
    args = parse_args(argv)

    # Configure logging
    import logging

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)

    # Parse steps
    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    try:
        result = run_pipeline(
            steps=steps,
            dry_run=args.dry_run,
            force=args.force,
        )
        return result.exit_code
    except ValueError as e:
        logger.error("Invalid configuration: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
