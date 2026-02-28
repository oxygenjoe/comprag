#!/usr/bin/env python3
"""Unit tests for scripts/run_pipeline.py.

Tests step discovery, --steps flag parsing, dry-run synthetic result generation
matching the v2 schema, and pipeline summary report format. All tests are pure
unit tests with no real data dependencies — downstream operations are mocked.

Run:
    python -m pytest tests/test_pipeline.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_pipeline import (
    ALL_STEPS,
    STEP_REGISTRY,
    SYNTH_DATASETS,
    SYNTH_MODELS,
    SYNTH_SAMPLES_PER_COMBO,
    SYNTH_SEEDS,
    Pipeline,
    PipelineResult,
    StepResult,
    generate_synthetic_results,
    main,
    parse_args,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Test 1: Step discovery — all expected steps present
# ---------------------------------------------------------------------------


class TestStepDiscovery:
    """ALL_STEPS and STEP_REGISTRY contain all expected pipeline steps."""

    EXPECTED_STEPS = [
        "download",
        "normalize",
        "validate_datasets",
        "build_index",
        "validate_index",
        "run",
        "aggregate",
        "visualize",
    ]

    def test_all_steps_list(self):
        """ALL_STEPS contains the expected 8 pipeline steps."""
        assert ALL_STEPS == self.EXPECTED_STEPS

    def test_step_count(self):
        """Exactly 8 steps are defined."""
        assert len(ALL_STEPS) == 8

    def test_step_registry_keys(self):
        """STEP_REGISTRY has an entry for each step in ALL_STEPS."""
        for step in ALL_STEPS:
            assert step in STEP_REGISTRY, f"Missing registry entry for step: {step}"

    def test_step_registry_callables(self):
        """Each step registry entry is callable."""
        for step_name, step_fn in STEP_REGISTRY.items():
            assert callable(step_fn), f"Step '{step_name}' is not callable"

    def test_step_order(self):
        """Steps are ordered: download -> normalize -> ... -> visualize."""
        assert ALL_STEPS[0] == "download"
        assert ALL_STEPS[-1] == "visualize"
        assert ALL_STEPS.index("run") < ALL_STEPS.index("aggregate")
        assert ALL_STEPS.index("aggregate") < ALL_STEPS.index("visualize")

    def test_no_duplicate_steps(self):
        """ALL_STEPS has no duplicates."""
        assert len(ALL_STEPS) == len(set(ALL_STEPS))


# ---------------------------------------------------------------------------
# Test 2: --steps flag parsing
# ---------------------------------------------------------------------------


class TestStepsParsing:
    """parse_args correctly handles the --steps flag."""

    def test_no_steps_flag(self):
        """Without --steps, args.steps is None (all steps)."""
        args = parse_args([])
        assert args.steps is None

    def test_single_step(self):
        """--steps with a single step name."""
        args = parse_args(["--steps", "download"])
        assert args.steps == "download"

    def test_comma_separated_steps(self):
        """--steps with comma-separated values."""
        args = parse_args(["--steps", "run,aggregate,visualize"])
        assert args.steps == "run,aggregate,visualize"

    def test_dry_run_flag(self):
        """--dry-run flag is parsed."""
        args = parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_force_flag(self):
        """--force flag is parsed."""
        args = parse_args(["--force"])
        assert args.force is True

    def test_verbose_flag(self):
        """--verbose / -v flag is parsed."""
        args = parse_args(["--verbose"])
        assert args.verbose is True
        args2 = parse_args(["-v"])
        assert args2.verbose is True

    def test_all_flags_combined(self):
        """All flags can be combined."""
        args = parse_args([
            "--dry-run", "--force", "--verbose",
            "--steps", "run,aggregate"
        ])
        assert args.dry_run is True
        assert args.force is True
        assert args.verbose is True
        assert args.steps == "run,aggregate"

    def test_pipeline_parses_comma_steps(self):
        """Pipeline constructor splits comma-separated steps correctly."""
        steps_str = "run,aggregate,visualize"
        steps_list = [s.strip() for s in steps_str.split(",") if s.strip()]
        pipeline = Pipeline(steps=steps_list, dry_run=True)
        assert pipeline.steps == ["run", "aggregate", "visualize"]

    def test_pipeline_invalid_step_raises(self):
        """Pipeline raises ValueError for unknown step names."""
        with pytest.raises(ValueError, match="Unknown step"):
            Pipeline(steps=["nonexistent_step"])

    def test_pipeline_default_all_steps(self):
        """Pipeline with steps=None defaults to ALL_STEPS."""
        pipeline = Pipeline(steps=None, dry_run=True)
        assert pipeline.steps == ALL_STEPS

    def test_main_parses_steps_and_splits(self):
        """main() correctly splits --steps comma string into a list."""
        # We mock run_pipeline to capture the steps argument
        with patch("scripts.run_pipeline.run_pipeline") as mock_run:
            mock_run.return_value = PipelineResult()
            main(["--dry-run", "--steps", "download,normalize"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs[1]["steps"] == ["download", "normalize"]


# ---------------------------------------------------------------------------
# Test 3: Dry-run synthetic result generation matches v2 schema
# ---------------------------------------------------------------------------


class TestSyntheticResults:
    """generate_synthetic_results produces v2-compliant JSONL."""

    def test_generates_jsonl_file(self, tmp_path):
        """Output file is created and non-empty."""
        output_path = generate_synthetic_results(tmp_path, seed=42)
        assert output_path.exists()
        assert output_path.suffix == ".jsonl"
        assert output_path.stat().st_size > 0

    def test_expected_record_count(self, tmp_path):
        """Number of records matches combinatoric expectation."""
        output_path = generate_synthetic_results(tmp_path, seed=42)
        with open(output_path) as f:
            records = [json.loads(line) for line in f]

        # Calculate expected count
        expected = 0
        for model, quant in SYNTH_MODELS:
            for dataset, subsets in SYNTH_DATASETS.items():
                for subset in subsets:
                    expected += len(SYNTH_SEEDS) * SYNTH_SAMPLES_PER_COMBO

        assert len(records) == expected

    def test_v2_required_keys(self, tmp_path):
        """Every record has all v2 required top-level keys."""
        output_path = generate_synthetic_results(tmp_path, seed=42)
        with open(output_path) as f:
            records = [json.loads(line) for line in f]

        required_keys = [
            "sample_id", "dataset", "subset", "query", "ground_truth",
            "response", "retrieved_chunks", "run_config", "perf",
            "metrics", "timestamp", "error",
        ]

        for i, record in enumerate(records):
            for key in required_keys:
                assert key in record, (
                    f"Record {i} missing v2 key: {key}"
                )

    def test_run_config_structure(self, tmp_path):
        """run_config block has model, quant, hardware, seed."""
        output_path = generate_synthetic_results(tmp_path, seed=42)
        with open(output_path) as f:
            record = json.loads(f.readline())

        rc = record["run_config"]
        assert "model" in rc
        assert "quant" in rc
        assert "hardware" in rc
        assert "seed" in rc
        assert isinstance(rc["model"], str)
        assert isinstance(rc["quant"], str)
        assert isinstance(rc["seed"], int)

    def test_perf_structure(self, tmp_path):
        """perf block has ttft_ms, total_tokens, tokens_per_sec, wall_clock_sec, vram_mb, gpu_temp."""
        output_path = generate_synthetic_results(tmp_path, seed=42)
        with open(output_path) as f:
            record = json.loads(f.readline())

        perf = record["perf"]
        perf_keys = ["ttft_ms", "total_tokens", "tokens_per_sec",
                      "wall_clock_sec", "vram_mb", "gpu_temp"]
        for key in perf_keys:
            assert key in perf, f"Missing perf key: {key}"
            assert isinstance(perf[key], (int, float)), (
                f"perf.{key} should be numeric, got {type(perf[key])}"
            )

    def test_metrics_structure(self, tmp_path):
        """metrics block has all six quality metrics."""
        output_path = generate_synthetic_results(tmp_path, seed=42)
        with open(output_path) as f:
            record = json.loads(f.readline())

        metrics = record["metrics"]
        metric_keys = [
            "faithfulness", "context_utilization", "self_knowledge",
            "noise_sensitivity", "answer_relevancy", "negative_rejection_rate",
        ]
        for key in metric_keys:
            assert key in metrics, f"Missing metric key: {key}"
            assert isinstance(metrics[key], float), (
                f"metrics.{key} should be float, got {type(metrics[key])}"
            )
            assert 0.0 <= metrics[key] <= 1.0, (
                f"metrics.{key}={metrics[key]} out of [0,1] range"
            )

    def test_retrieved_chunks_structure(self, tmp_path):
        """retrieved_chunks is a list of dicts with chunk_id, text, score."""
        output_path = generate_synthetic_results(tmp_path, seed=42)
        with open(output_path) as f:
            record = json.loads(f.readline())

        chunks = record["retrieved_chunks"]
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "score" in chunk
            assert isinstance(chunk["score"], float)

    def test_sample_id_format(self, tmp_path):
        """sample_id follows {dataset}_{subset}_{index:04d} format."""
        output_path = generate_synthetic_results(tmp_path, seed=42)
        with open(output_path) as f:
            record = json.loads(f.readline())

        sample_id = record["sample_id"]
        assert isinstance(sample_id, str)
        parts = sample_id.rsplit("_", 1)
        assert len(parts) == 2
        assert parts[1].isdigit()
        assert len(parts[1]) == 4  # zero-padded

    def test_error_is_none(self, tmp_path):
        """Synthetic records have error=None (no errors in synthetic mode)."""
        output_path = generate_synthetic_results(tmp_path, seed=42)
        with open(output_path) as f:
            records = [json.loads(line) for line in f]

        for record in records:
            assert record["error"] is None

    def test_seed_reproducibility(self, tmp_path):
        """Same seed produces identical JSONL output."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        path1 = generate_synthetic_results(dir1, seed=77)
        path2 = generate_synthetic_results(dir2, seed=77)

        text1 = path1.read_text()
        text2 = path2.read_text()
        # Timestamps will differ; compare everything else
        records1 = [json.loads(line) for line in text1.strip().split("\n")]
        records2 = [json.loads(line) for line in text2.strip().split("\n")]
        assert len(records1) == len(records2)
        for r1, r2 in zip(records1, records2):
            # Compare all fields except timestamp (which uses datetime.now)
            r1.pop("timestamp")
            r2.pop("timestamp")
            assert r1 == r2

    def test_jsonl_is_valid_json_per_line(self, tmp_path):
        """Each line is valid JSON (no trailing commas, valid structure)."""
        output_path = generate_synthetic_results(tmp_path, seed=42)
        with open(output_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON on line {line_num}: {e}")


# ---------------------------------------------------------------------------
# Test 4: Pipeline summary report format
# ---------------------------------------------------------------------------


class TestPipelineSummaryReport:
    """Pipeline run produces a structured summary with correct format."""

    def test_step_result_dataclass(self):
        """StepResult has expected fields with correct defaults."""
        sr = StepResult(name="test_step")
        assert sr.name == "test_step"
        assert sr.status == "pending"
        assert sr.elapsed_sec == 0.0
        assert sr.message == ""
        assert sr.error is None

    def test_pipeline_result_properties(self):
        """PipelineResult computes succeeded/failed/skipped correctly."""
        pr = PipelineResult()
        pr.steps = [
            StepResult(name="s1", status="success"),
            StepResult(name="s2", status="success"),
            StepResult(name="s3", status="skipped"),
            StepResult(name="s4", status="error", error="something broke"),
        ]
        assert pr.succeeded == 2
        assert pr.failed == 1
        assert pr.skipped == 1
        assert pr.exit_code == 1  # has at least one failure

    def test_pipeline_result_exit_code_success(self):
        """PipelineResult.exit_code is 0 when no errors."""
        pr = PipelineResult()
        pr.steps = [
            StepResult(name="s1", status="success"),
            StepResult(name="s2", status="skipped"),
        ]
        assert pr.exit_code == 0

    def test_pipeline_result_exit_code_all_skipped(self):
        """PipelineResult.exit_code is 0 when all skipped (no errors)."""
        pr = PipelineResult()
        pr.steps = [
            StepResult(name="s1", status="skipped"),
            StepResult(name="s2", status="skipped"),
        ]
        assert pr.exit_code == 0

    def test_pipeline_result_empty(self):
        """Empty PipelineResult has exit_code 0."""
        pr = PipelineResult()
        assert pr.succeeded == 0
        assert pr.failed == 0
        assert pr.skipped == 0
        assert pr.exit_code == 0

    def test_dry_run_pipeline_runs_all_steps(self):
        """Dry-run pipeline executes all steps (skips data steps, runs 'run')."""
        # Patch all downstream imports that steps might use
        with patch.dict("sys.modules", {
            "scripts.download_datasets": MagicMock(),
            "scripts.normalize_datasets": MagicMock(),
            "scripts.validate_datasets": MagicMock(),
            "scripts.build_index": MagicMock(),
            "scripts.validate_index": MagicMock(),
            "scripts.visualize_results": MagicMock(),
            "cumrag.aggregator": MagicMock(),
        }):
            result = run_pipeline(dry_run=True, force=True)

        assert isinstance(result, PipelineResult)
        assert result.dry_run is True
        assert len(result.steps) == 8

        # In dry-run, download/normalize/validate_datasets/build_index/validate_index
        # are skipped; run generates synthetic data
        skipped_steps = [s for s in result.steps if s.status == "skipped"]
        assert len(skipped_steps) >= 5, (
            f"Expected >=5 skipped steps in dry-run, got {len(skipped_steps)}: "
            f"{[s.name for s in skipped_steps]}"
        )

    def test_pipeline_subset_steps(self):
        """Pipeline with --steps runs only specified steps."""
        pipeline = Pipeline(steps=["download", "normalize"], dry_run=True)
        assert len(pipeline.steps) == 2
        assert pipeline.steps == ["download", "normalize"]

    def test_step_result_timing(self):
        """StepResult records elapsed time."""
        sr = StepResult(name="timed_step", status="success", elapsed_sec=1.234)
        assert sr.elapsed_sec == pytest.approx(1.234)

    def test_pipeline_prints_summary(self, capsys):
        """Pipeline.run() prints a summary with step statuses."""
        pipeline = Pipeline(steps=["download"], dry_run=True)
        result = pipeline.run()

        captured = capsys.readouterr()
        assert "Pipeline Summary" in captured.out
        assert "download" in captured.out

    def test_main_returns_exit_code(self):
        """main() returns integer exit code."""
        with patch("scripts.run_pipeline.run_pipeline") as mock_run:
            mock_run.return_value = PipelineResult()
            exit_code = main(["--dry-run", "--steps", "download"])
            assert isinstance(exit_code, int)
            assert exit_code == 0

    def test_main_invalid_step_returns_1(self):
        """main() returns 1 for invalid step names."""
        exit_code = main(["--steps", "bogus_step"])
        assert exit_code == 1
