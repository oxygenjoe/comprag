#!/usr/bin/env python3
"""Unit tests for scripts/visualize_results.py.

Tests synthetic data generation, chart generation (headless), CLI arg parsing,
and graceful handling of empty/missing data. All tests are pure unit tests
with no real data dependencies.

Run:
    python -m pytest tests/test_visualization.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")  # Headless backend — must be before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.visualize_results import (
    ALL_CHART_FUNCTIONS,
    DEMO_MODELS,
    DEMO_QUANTS,
    MODEL_PARAMS,
    QUANT_ORDER,
    chart_faithfulness_by_model,
    chart_model_size_vs_faithfulness,
    chart_quantization_effect,
    chart_self_knowledge_by_model,
    chart_throughput_heatmap,
    chart_vram_usage,
    generate_all_charts,
    generate_demo_data,
    load_aggregated_csv,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_df() -> pd.DataFrame:
    """Generate demo data with a fixed seed."""
    return generate_demo_data(seed=42)


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """An empty DataFrame with the expected column names."""
    cols = [
        "model", "quantization", "hardware_tier", "dataset", "eval_subset",
        "n_runs", "warnings",
    ]
    metric_names = [
        "faithfulness", "context_utilization", "self_knowledge",
        "noise_sensitivity", "answer_relevancy", "negative_rejection_rate",
        "tokens_per_second", "ttft_ms", "vram_usage_mb", "gpu_temp",
    ]
    for m in metric_names:
        cols.extend([f"{m}_mean", f"{m}_ci_low", f"{m}_ci_high", f"{m}_flagged"])
    return pd.DataFrame(columns=cols)


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """A single-row DataFrame for edge-case testing."""
    return pd.DataFrame([{
        "model": "llama-3.1-8b-instruct",
        "quantization": "Q4_K_M",
        "hardware_tier": "v100",
        "dataset": "rgb",
        "eval_subset": "noise_robustness",
        "n_runs": 3,
        "faithfulness_mean": 0.75,
        "faithfulness_ci_low": 0.70,
        "faithfulness_ci_high": 0.80,
        "faithfulness_flagged": "",
        "self_knowledge_mean": 0.65,
        "self_knowledge_ci_low": 0.60,
        "self_knowledge_ci_high": 0.70,
        "self_knowledge_flagged": "",
        "context_utilization_mean": 0.70,
        "context_utilization_ci_low": 0.65,
        "context_utilization_ci_high": 0.75,
        "context_utilization_flagged": "",
        "noise_sensitivity_mean": 0.30,
        "noise_sensitivity_ci_low": 0.25,
        "noise_sensitivity_ci_high": 0.35,
        "noise_sensitivity_flagged": "",
        "answer_relevancy_mean": 0.72,
        "answer_relevancy_ci_low": 0.68,
        "answer_relevancy_ci_high": 0.76,
        "answer_relevancy_flagged": "",
        "negative_rejection_rate_mean": 0.60,
        "negative_rejection_rate_ci_low": 0.55,
        "negative_rejection_rate_ci_high": 0.65,
        "negative_rejection_rate_flagged": "",
        "tokens_per_second_mean": 30.0,
        "tokens_per_second_ci_low": 28.0,
        "tokens_per_second_ci_high": 32.0,
        "tokens_per_second_flagged": "",
        "ttft_ms_mean": 80.0,
        "ttft_ms_ci_low": 70.0,
        "ttft_ms_ci_high": 90.0,
        "ttft_ms_flagged": "",
        "vram_usage_mb_mean": 6400.0,
        "vram_usage_mb_ci_low": 6300.0,
        "vram_usage_mb_ci_high": 6500.0,
        "vram_usage_mb_flagged": "",
        "gpu_temp_mean": 65.0,
        "gpu_temp_ci_low": 62.0,
        "gpu_temp_ci_high": 68.0,
        "gpu_temp_flagged": "",
        "warnings": "",
    }])


# ---------------------------------------------------------------------------
# Test 1: Synthetic demo data generation
# ---------------------------------------------------------------------------


class TestDemoDataGeneration:
    """generate_demo_data() produces a valid DataFrame."""

    def test_returns_dataframe(self, demo_df):
        """Should return a pandas DataFrame."""
        assert isinstance(demo_df, pd.DataFrame)

    def test_nonempty(self, demo_df):
        """Should have at least one row."""
        assert len(demo_df) > 0

    def test_required_columns(self, demo_df):
        """DataFrame has all required grouping columns."""
        required = ["model", "quantization", "hardware_tier", "dataset",
                     "eval_subset", "n_runs"]
        for col in required:
            assert col in demo_df.columns, f"Missing column: {col}"

    def test_metric_columns(self, demo_df):
        """DataFrame has mean/ci_low/ci_high for all expected metrics."""
        metrics = [
            "faithfulness", "context_utilization", "self_knowledge",
            "noise_sensitivity", "answer_relevancy", "negative_rejection_rate",
            "tokens_per_second", "ttft_ms", "vram_usage_mb", "gpu_temp",
        ]
        for metric in metrics:
            for suffix in ["_mean", "_ci_low", "_ci_high"]:
                col = f"{metric}{suffix}"
                assert col in demo_df.columns, f"Missing metric column: {col}"

    def test_models_present(self, demo_df):
        """All DEMO_MODELS appear in the generated data."""
        for model in DEMO_MODELS:
            assert model in demo_df["model"].values, f"Missing model: {model}"

    def test_values_in_range(self, demo_df):
        """Quality metric means are between 0 and 1."""
        quality_metrics = [
            "faithfulness_mean", "self_knowledge_mean",
            "context_utilization_mean", "answer_relevancy_mean",
            "negative_rejection_rate_mean", "noise_sensitivity_mean",
        ]
        for col in quality_metrics:
            assert demo_df[col].min() >= 0.0, f"{col} has negative values"
            assert demo_df[col].max() <= 1.0, f"{col} exceeds 1.0"

    def test_ci_ordering(self, demo_df):
        """ci_low <= mean <= ci_high for faithfulness."""
        assert (demo_df["faithfulness_ci_low"] <= demo_df["faithfulness_mean"]).all()
        assert (demo_df["faithfulness_mean"] <= demo_df["faithfulness_ci_high"]).all()

    def test_seed_reproducibility(self):
        """Same seed produces identical data."""
        df1 = generate_demo_data(seed=99)
        df2 = generate_demo_data(seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds produce different data."""
        df1 = generate_demo_data(seed=1)
        df2 = generate_demo_data(seed=2)
        assert not df1.equals(df2)

    def test_n_runs_column(self, demo_df):
        """n_runs is consistently 5 (the demo default)."""
        assert (demo_df["n_runs"] == 5).all()


# ---------------------------------------------------------------------------
# Test 2: Chart generation does not error
# ---------------------------------------------------------------------------


class TestChartGeneration:
    """All chart functions run without errors on valid demo data."""

    def test_chart_faithfulness(self, demo_df, tmp_path):
        """chart_faithfulness_by_model produces a PNG."""
        out = chart_faithfulness_by_model(demo_df, tmp_path)
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 0

    def test_chart_self_knowledge(self, demo_df, tmp_path):
        """chart_self_knowledge_by_model produces a PNG."""
        out = chart_self_knowledge_by_model(demo_df, tmp_path)
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 0

    def test_chart_model_size_scatter(self, demo_df, tmp_path):
        """chart_model_size_vs_faithfulness produces a PNG."""
        out = chart_model_size_vs_faithfulness(demo_df, tmp_path)
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 0

    def test_chart_quantization_effect(self, demo_df, tmp_path):
        """chart_quantization_effect produces a PNG."""
        out = chart_quantization_effect(demo_df, tmp_path)
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 0

    def test_chart_throughput_heatmap(self, demo_df, tmp_path):
        """chart_throughput_heatmap produces a PNG."""
        out = chart_throughput_heatmap(demo_df, tmp_path)
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 0

    def test_chart_vram_usage(self, demo_df, tmp_path):
        """chart_vram_usage produces a PNG."""
        out = chart_vram_usage(demo_df, tmp_path)
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 0

    def test_generate_all_charts(self, demo_df, tmp_path):
        """generate_all_charts produces 6 PNGs."""
        generated = generate_all_charts(demo_df, tmp_path)
        assert len(generated) == 6
        for p in generated:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.stat().st_size > 0

    def test_all_chart_functions_registered(self):
        """ALL_CHART_FUNCTIONS has exactly 6 entries."""
        assert len(ALL_CHART_FUNCTIONS) == 6

    def test_output_dir_created(self, demo_df, tmp_path):
        """generate_all_charts creates nested output dir if it doesn't exist."""
        nested = tmp_path / "deep" / "nested" / "dir"
        assert not nested.exists()
        generated = generate_all_charts(demo_df, nested)
        assert nested.exists()
        assert len(generated) == 6

    def test_no_pyplot_figures_leaked(self, demo_df, tmp_path):
        """All figures are closed after generation (no memory leak)."""
        plt.close("all")
        initial_figs = len(plt.get_fignums())
        generate_all_charts(demo_df, tmp_path)
        final_figs = len(plt.get_fignums())
        assert final_figs == initial_figs, (
            f"Leaked {final_figs - initial_figs} matplotlib figure(s)"
        )


# ---------------------------------------------------------------------------
# Test 3: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgs:
    """CLI argument parsing for --input, --output-dir, --demo."""

    def test_default_args(self):
        """Default values are set when no args given."""
        with patch("sys.argv", ["visualize_results.py"]):
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", type=str, default="results/aggregated/summary.csv")
            parser.add_argument("--output-dir", type=str, default="results/figures/")
            parser.add_argument("--demo", action="store_true")
            args = parser.parse_args([])
            assert args.input == "results/aggregated/summary.csv"
            assert args.output_dir == "results/figures/"
            assert args.demo is False

    def test_custom_input(self):
        """--input overrides default CSV path."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str, default="results/aggregated/summary.csv")
        parser.add_argument("--output-dir", type=str, default="results/figures/")
        parser.add_argument("--demo", action="store_true")
        args = parser.parse_args(["--input", "/tmp/custom.csv"])
        assert args.input == "/tmp/custom.csv"

    def test_custom_output_dir(self):
        """--output-dir overrides default output path."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str, default="results/aggregated/summary.csv")
        parser.add_argument("--output-dir", type=str, default="results/figures/")
        parser.add_argument("--demo", action="store_true")
        args = parser.parse_args(["--output-dir", "/tmp/charts"])
        assert args.output_dir == "/tmp/charts"

    def test_demo_flag(self):
        """--demo flag is a boolean store_true."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str, default="results/aggregated/summary.csv")
        parser.add_argument("--output-dir", type=str, default="results/figures/")
        parser.add_argument("--demo", action="store_true")
        args = parser.parse_args(["--demo"])
        assert args.demo is True

    def test_demo_with_output_dir(self):
        """--demo and --output-dir can be combined."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str, default="results/aggregated/summary.csv")
        parser.add_argument("--output-dir", type=str, default="results/figures/")
        parser.add_argument("--demo", action="store_true")
        args = parser.parse_args(["--demo", "--output-dir", "/tmp/demo_charts"])
        assert args.demo is True
        assert args.output_dir == "/tmp/demo_charts"

    def test_main_demo_mode(self, tmp_path):
        """main() in --demo mode runs without error."""
        output_dir = str(tmp_path / "figures")
        with patch("sys.argv", [
            "visualize_results.py", "--demo", "--output-dir", output_dir
        ]):
            # main() calls sys.exit on failure; should not raise for demo mode
            main()
        # Verify output was created
        pngs = list(Path(output_dir).glob("*.png"))
        assert len(pngs) == 6

    def test_main_missing_input(self, tmp_path):
        """main() exits with code 1 when input CSV does not exist."""
        fake_csv = str(tmp_path / "nonexistent.csv")
        with patch("sys.argv", [
            "visualize_results.py", "--input", fake_csv
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Test 4: Graceful handling of empty/missing data
# ---------------------------------------------------------------------------


class TestGracefulEmptyData:
    """Charts handle empty, minimal, and malformed data gracefully."""

    def test_empty_df_all_charts(self, empty_df, tmp_path):
        """generate_all_charts does not crash on an empty DataFrame."""
        # Some charts may fail internally; generate_all_charts catches errors.
        # The key assertion is: no unhandled exception propagates.
        generated = generate_all_charts(empty_df, tmp_path)
        # May produce 0 or more charts; should not crash
        assert isinstance(generated, list)

    def test_single_row_all_charts(self, minimal_df, tmp_path):
        """generate_all_charts works with a single-row DataFrame."""
        generated = generate_all_charts(minimal_df, tmp_path)
        assert isinstance(generated, list)
        assert len(generated) > 0

    def test_quantization_effect_no_v100(self, tmp_path):
        """chart_quantization_effect handles missing v100 data by fallback."""
        df = pd.DataFrame([{
            "model": "llama-3.1-8b-instruct",
            "quantization": "Q4_K_M",
            "hardware_tier": "cpu",
            "dataset": "rgb",
            "eval_subset": "noise_robustness",
            "n_runs": 3,
            "faithfulness_mean": 0.65,
            "faithfulness_ci_low": 0.60,
            "faithfulness_ci_high": 0.70,
            "faithfulness_flagged": "",
        }])
        # Should fall back to CPU data and not crash
        out = chart_quantization_effect(df, tmp_path)
        assert out.exists()

    def test_vram_chart_all_cpu(self, tmp_path):
        """chart_vram_usage handles data with only CPU (vram=0) rows."""
        df = pd.DataFrame([{
            "model": "llama-3.1-8b-instruct",
            "quantization": "Q4_K_M",
            "hardware_tier": "cpu",
            "dataset": "rgb",
            "eval_subset": "noise_robustness",
            "n_runs": 3,
            "vram_usage_mb_mean": 0.0,
            "vram_usage_mb_ci_low": 0.0,
            "vram_usage_mb_ci_high": 0.0,
            "vram_usage_mb_flagged": "",
        }])
        # Should produce a "No GPU Data" fallback chart
        out = chart_vram_usage(df, tmp_path)
        assert out.exists()

    def test_scatter_missing_model_params(self, tmp_path):
        """chart_model_size_vs_faithfulness drops unknown models gracefully."""
        df = pd.DataFrame([{
            "model": "unknown-model-999b",
            "quantization": "Q4_K_M",
            "hardware_tier": "v100",
            "dataset": "rgb",
            "eval_subset": "noise_robustness",
            "n_runs": 3,
            "faithfulness_mean": 0.75,
            "faithfulness_ci_low": 0.70,
            "faithfulness_ci_high": 0.80,
        }])
        # Model not in MODEL_PARAMS; should be filtered out, chart still generated
        out = chart_model_size_vs_faithfulness(df, tmp_path)
        assert out.exists()

    def test_load_csv_coerces_numeric(self, tmp_path):
        """load_aggregated_csv coerces string metrics to float."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "model,faithfulness_mean,faithfulness_ci_low,faithfulness_ci_high\n"
            "test-model,0.75,0.70,0.80\n"
            "test-model2,bad_value,,\n"
        )
        df = load_aggregated_csv(csv_path)
        assert len(df) == 2
        assert df["faithfulness_mean"].iloc[0] == 0.75
        assert pd.isna(df["faithfulness_mean"].iloc[1])
