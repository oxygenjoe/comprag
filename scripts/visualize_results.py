#!/usr/bin/env python3
"""Publication-quality visualization of CompRAG evaluation results.

Generates six chart types from aggregated CSV data (v2 schema):
  1. Faithfulness by model (grouped bar)
  2. Self-knowledge by model (grouped bar)
  3. Model size vs faithfulness scatter with trend line
  4. Quantization effect on faithfulness per model
  5. Throughput heatmap (model x hardware)
  6. VRAM usage bar chart

All charts include bootstrap CI error bars where applicable, use a consistent
seaborn color palette, and are saved as 300 DPI PNGs.

CLI usage:
    python scripts/visualize_results.py --input results/aggregated/summary.csv --output-dir results/figures/
    python scripts/visualize_results.py --demo  # Generate from synthetic data

Importable:
    from scripts.visualize_results import generate_all_charts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

DPI = 300
FIGSIZE_BAR = (12, 6)
FIGSIZE_SCATTER = (10, 7)
FIGSIZE_HEATMAP = (12, 7)
FIGSIZE_VRAM = (10, 6)

# Consistent palette across all charts
PALETTE_NAME = "Set2"
FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK = 10

# Model parameter sizes for scatter plot (billions)
MODEL_PARAMS: dict[str, float] = {
    "qwen2.5-14b-instruct": 14.0,
    "phi-4-14b": 14.0,
    "mistral-nemo-12b-instruct": 12.0,
    "llama-3.1-8b-instruct": 8.0,
    "qwen2.5-7b-instruct": 7.0,
    "gemma-2-9b-instruct": 9.0,
    "glm-4-9b-chat": 9.0,
    "smollm2-1.7b-instruct": 1.7,
}


def _apply_style() -> None:
    """Set global matplotlib/seaborn aesthetics."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_TICK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_aggregated_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load an aggregated summary CSV into a DataFrame.

    Expects columns matching the v2 aggregated schema output by
    comprag.aggregator: model, quantization, hardware_tier, dataset,
    eval_subset, n_runs, and {metric}_{mean,ci_low,ci_high,flagged} columns.

    Numeric columns are coerced; empty strings become NaN.
    """
    df = pd.read_csv(csv_path)

    # Coerce metric columns to numeric
    numeric_suffixes = ("_mean", "_ci_low", "_ci_high")
    for col in df.columns:
        if any(col.endswith(s) for s in numeric_suffixes):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Synthetic demo data
# ---------------------------------------------------------------------------

# Quantization ordering for consistent axis labeling
QUANT_ORDER = ["Q4_K_M", "Q8_0", "FP16"]

DEMO_MODELS = [
    "qwen2.5-14b-instruct",
    "phi-4-14b",
    "mistral-nemo-12b-instruct",
    "llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct",
    "gemma-2-9b-instruct",
    "glm-4-9b-chat",
    "smollm2-1.7b-instruct",
]

DEMO_QUANTS: dict[str, list[str]] = {
    "qwen2.5-14b-instruct": ["Q4_K_M", "Q8_0", "FP16"],
    "phi-4-14b": ["Q4_K_M", "Q8_0", "FP16"],
    "mistral-nemo-12b-instruct": ["Q4_K_M", "Q8_0"],
    "llama-3.1-8b-instruct": ["Q4_K_M", "Q8_0", "FP16"],
    "qwen2.5-7b-instruct": ["Q4_K_M", "Q8_0"],
    "gemma-2-9b-instruct": ["Q4_K_M", "Q8_0"],
    "glm-4-9b-chat": ["Q4_K_M"],
    "smollm2-1.7b-instruct": ["Q8_0", "FP16"],
}

DEMO_HARDWARE = ["v100", "cpu", "1660s"]

# VRAM sizes (GB) for demo data keyed by model+quant
DEMO_VRAM_GB: dict[str, dict[str, float]] = {
    "qwen2.5-14b-instruct": {"Q4_K_M": 10.2, "Q8_0": 17.1, "FP16": 30.5},
    "phi-4-14b": {"Q4_K_M": 9.8, "Q8_0": 16.2, "FP16": 29.0},
    "mistral-nemo-12b-instruct": {"Q4_K_M": 8.8, "Q8_0": 14.5},
    "llama-3.1-8b-instruct": {"Q4_K_M": 6.2, "Q8_0": 10.0, "FP16": 17.5},
    "qwen2.5-7b-instruct": {"Q4_K_M": 6.0, "Q8_0": 9.6},
    "gemma-2-9b-instruct": {"Q4_K_M": 7.2, "Q8_0": 11.5},
    "glm-4-9b-chat": {"Q4_K_M": 6.9},
    "smollm2-1.7b-instruct": {"Q8_0": 2.5, "FP16": 4.0},
}


def generate_demo_data(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic aggregated data matching the v2 CSV schema.

    Produces plausible scores based on model size and quantization level.
    Larger models score higher; higher-precision quants score slightly higher.
    Error bars are synthetic but proportional.
    """
    rng = np.random.default_rng(seed)

    rows: list[dict] = []

    for model in DEMO_MODELS:
        params_b = MODEL_PARAMS[model]
        for quant in DEMO_QUANTS[model]:
            for hw in DEMO_HARDWARE:
                # Skip infeasible combos
                if hw == "1660s" and params_b > 9:
                    continue
                if hw == "cpu":
                    # CPU only runs Q4_K_M
                    if quant != "Q4_K_M":
                        continue

                # Base quality scales with log(params)
                base_quality = 0.45 + 0.15 * np.log2(params_b / 1.7)
                # Quant bonus
                quant_bonus = {"Q4_K_M": 0.0, "Q8_0": 0.02, "FP16": 0.04}.get(
                    quant, 0.0
                )
                base_quality = min(base_quality + quant_bonus, 0.95)

                # Per-metric noise
                faith_mean = base_quality + rng.normal(0, 0.02)
                faith_mean = float(np.clip(faith_mean, 0.1, 0.98))
                sk_mean = base_quality * 0.85 + rng.normal(0, 0.02)
                sk_mean = float(np.clip(sk_mean, 0.1, 0.95))
                cu_mean = base_quality * 0.9 + rng.normal(0, 0.02)
                cu_mean = float(np.clip(cu_mean, 0.1, 0.95))
                ns_mean = 0.3 + rng.normal(0, 0.03)
                ns_mean = float(np.clip(ns_mean, 0.05, 0.6))
                ar_mean = base_quality * 0.95 + rng.normal(0, 0.02)
                ar_mean = float(np.clip(ar_mean, 0.1, 0.98))
                nr_mean = base_quality * 0.8 + rng.normal(0, 0.03)
                nr_mean = float(np.clip(nr_mean, 0.1, 0.95))

                # Throughput depends on hardware and model size
                if hw == "v100":
                    tps_base = 45 - params_b * 1.5
                elif hw == "1660s":
                    tps_base = 15 - params_b * 0.8
                else:  # cpu
                    tps_base = 8 - params_b * 0.3
                # Higher quant = slower
                quant_tps_mult = {"Q4_K_M": 1.0, "Q8_0": 0.75, "FP16": 0.5}.get(
                    quant, 1.0
                )
                tps_mean = max(float(tps_base * quant_tps_mult + rng.normal(0, 1)), 0.5)

                # TTFT
                ttft_mean = float(50 + params_b * 5 + rng.normal(0, 10))
                if hw == "cpu":
                    ttft_mean *= 4

                # VRAM
                vram_mb = DEMO_VRAM_GB.get(model, {}).get(quant, 5.0) * 1024
                if hw == "cpu":
                    vram_mb = 0  # No GPU memory on CPU tier

                # CI half-widths (3-8% of mean for quality, wider for perf)
                def _ci(mean: float, pct: float = 0.05) -> tuple[float, float]:
                    half = abs(mean) * pct * (1 + rng.uniform(0, 0.5))
                    return (mean - half, mean + half)

                def _row_metrics(
                    metric: str, mean: float, pct: float = 0.05
                ) -> dict[str, object]:
                    low, high = _ci(mean, pct)
                    return {
                        f"{metric}_mean": round(mean, 4),
                        f"{metric}_ci_low": round(low, 4),
                        f"{metric}_ci_high": round(high, 4),
                        f"{metric}_flagged": "",
                    }

                row: dict = {
                    "model": model,
                    "quantization": quant,
                    "hardware_tier": hw,
                    "dataset": "rgb",
                    "eval_subset": "noise_robustness",
                    "n_runs": 5,
                }
                row.update(_row_metrics("faithfulness", faith_mean))
                row.update(_row_metrics("context_utilization", cu_mean))
                row.update(_row_metrics("self_knowledge", sk_mean))
                row.update(_row_metrics("noise_sensitivity", ns_mean))
                row.update(_row_metrics("answer_relevancy", ar_mean))
                row.update(_row_metrics("negative_rejection_rate", nr_mean))
                row.update(_row_metrics("tokens_per_second", tps_mean, 0.08))
                row.update(_row_metrics("ttft_ms", ttft_mean, 0.10))
                row.update(_row_metrics("vram_usage_mb", vram_mb, 0.02))
                row.update(_row_metrics("gpu_temp", 65 + rng.normal(0, 5), 0.03))
                row["warnings"] = ""

                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------


def _short_model_name(name: str) -> str:
    """Shorten model name for chart labels."""
    mapping = {
        "qwen2.5-14b-instruct": "Qwen2.5-14B",
        "phi-4-14b": "Phi-4-14B",
        "mistral-nemo-12b-instruct": "Mistral-12B",
        "llama-3.1-8b-instruct": "Llama3.1-8B",
        "qwen2.5-7b-instruct": "Qwen2.5-7B",
        "gemma-2-9b-instruct": "Gemma2-9B",
        "glm-4-9b-chat": "GLM4-9B",
        "smollm2-1.7b-instruct": "SmolLM2-1.7B",
    }
    return mapping.get(name, name)


def _sort_models_by_params(models: list[str]) -> list[str]:
    """Sort model names by parameter count (descending)."""
    return sorted(models, key=lambda m: MODEL_PARAMS.get(m, 0), reverse=True)


def _grouped_bar_chart(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    """Shared helper for grouped bar charts by model and hardware tier.

    Aggregates *metric* across quantizations/datasets, then draws a grouped
    bar chart with bootstrap CI error bars — one group per model, one bar per
    hardware tier.

    Args:
        df: DataFrame with v2 aggregated schema columns.
        output_dir: Directory to write the output PNG.
        metric: Base metric name (e.g. ``"faithfulness"``).
        title: Chart title.
        ylabel: Y-axis label.
        filename: Output PNG filename (e.g. ``"faithfulness_by_model.png"``).

    Returns:
        Path to the saved PNG.
    """
    col_mean = f"{metric}_mean"
    col_low = f"{metric}_ci_low"
    col_high = f"{metric}_ci_high"

    # Aggregate across quants/datasets: take mean per model x hardware
    grouped = (
        df.groupby(["model", "hardware_tier"])
        .agg({col_mean: "mean", col_low: "mean", col_high: "mean"})
        .reset_index()
    )

    models = _sort_models_by_params(grouped["model"].unique().tolist())
    hw_tiers = sorted(grouped["hardware_tier"].unique().tolist())
    palette = sns.color_palette(PALETTE_NAME, len(hw_tiers))

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    n_hw = len(hw_tiers)
    bar_width = 0.8 / max(n_hw, 1)
    x = np.arange(len(models))

    for i, hw in enumerate(hw_tiers):
        hw_data = grouped[grouped["hardware_tier"] == hw]
        means = []
        err_low = []
        err_high = []
        for model in models:
            row = hw_data[hw_data["model"] == model]
            if len(row) > 0:
                m = row[col_mean].values[0]
                lo = row[col_low].values[0]
                hi = row[col_high].values[0]
                means.append(m)
                err_low.append(m - lo)
                err_high.append(hi - m)
            else:
                means.append(0)
                err_low.append(0)
                err_high.append(0)

        offset = (i - n_hw / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=[err_low, err_high],
            capsize=3,
            label=hw,
            color=palette[i],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model_name(m) for m in models], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Hardware Tier", loc="lower right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    out = output_dir / filename
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_faithfulness_by_model(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Chart 1: Faithfulness by model, grouped by hardware tier.

    Grouped bar chart with bootstrap CI error bars.
    """
    return _grouped_bar_chart(
        df,
        output_dir,
        metric="faithfulness",
        title="Faithfulness by Model and Hardware Tier",
        ylabel="Faithfulness Score",
        filename="faithfulness_by_model.png",
    )


def chart_self_knowledge_by_model(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Chart 2: Self-knowledge by model, grouped by hardware tier.

    Grouped bar chart with bootstrap CI error bars.
    """
    return _grouped_bar_chart(
        df,
        output_dir,
        metric="self_knowledge",
        title="Self-Knowledge by Model and Hardware Tier",
        ylabel="Self-Knowledge Score",
        filename="self_knowledge_by_model.png",
    )


def chart_model_size_vs_faithfulness(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Chart 3: Model size (params) vs faithfulness scatter with trend line.

    Each point is a model+quant+hardware combo. Points are colored by hardware
    tier. A polynomial trend line (degree 2) is overlaid.
    """
    col_mean = "faithfulness_mean"

    plot_df = df[df[col_mean].notna()].copy()
    plot_df["params_b"] = plot_df["model"].map(MODEL_PARAMS)
    plot_df = plot_df.dropna(subset=["params_b"])

    hw_tiers = sorted(plot_df["hardware_tier"].unique().tolist())
    palette = sns.color_palette(PALETTE_NAME, len(hw_tiers))
    hw_colors = dict(zip(hw_tiers, palette))

    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)

    for hw in hw_tiers:
        subset = plot_df[plot_df["hardware_tier"] == hw]
        ax.scatter(
            subset["params_b"],
            subset[col_mean],
            label=hw,
            color=hw_colors[hw],
            s=80,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    # Trend line across all data
    x_all = plot_df["params_b"].values
    y_all = plot_df[col_mean].values
    if len(x_all) >= 3:
        coeffs = np.polyfit(x_all, y_all, 2)
        x_smooth = np.linspace(x_all.min(), x_all.max(), 100)
        y_smooth = np.polyval(coeffs, x_smooth)
        ax.plot(
            x_smooth,
            y_smooth,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Trend (poly-2)",
            zorder=2,
        )

    ax.set_xlabel("Model Size (Billion Parameters)")
    ax.set_ylabel("Faithfulness Score")
    ax.set_title("Model Size vs Faithfulness")
    ax.legend(title="Hardware Tier")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(x_all) * 1.1 if len(x_all) > 0 else 16)

    out = output_dir / "model_size_vs_faithfulness.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_quantization_effect(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Chart 4: Quantization effect on faithfulness per model.

    Line plot with error bars showing how faithfulness changes across quant
    levels for each model. Only V100 hardware to isolate the quant variable.
    """
    col_mean = "faithfulness_mean"
    col_low = "faithfulness_ci_low"
    col_high = "faithfulness_ci_high"

    # Filter to v100 to isolate quantization effect
    v100_df = df[df["hardware_tier"] == "v100"].copy()
    if v100_df.empty:
        # Fallback: use whatever hardware has the most data
        hw_counts = df["hardware_tier"].value_counts()
        if hw_counts.empty:
            # Create an empty chart
            fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
            ax.set_title("Quantization Effect on Faithfulness (No Data)")
            out = output_dir / "quantization_effect.png"
            fig.savefig(out)
            plt.close(fig)
            return out
        v100_df = df[df["hardware_tier"] == hw_counts.index[0]].copy()

    # Average across datasets/subsets
    grouped = (
        v100_df.groupby(["model", "quantization"])
        .agg({col_mean: "mean", col_low: "mean", col_high: "mean"})
        .reset_index()
    )

    models = _sort_models_by_params(grouped["model"].unique().tolist())
    palette = sns.color_palette(PALETTE_NAME, len(models))

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)

    # Map quant to x position
    quant_order = [q for q in QUANT_ORDER if q in grouped["quantization"].values]
    quant_x = {q: i for i, q in enumerate(quant_order)}

    for idx, model in enumerate(models):
        m_data = grouped[grouped["model"] == model]
        m_data = m_data[m_data["quantization"].isin(quant_order)]
        if m_data.empty:
            continue

        xs = [quant_x[q] for q in m_data["quantization"]]
        ys = m_data[col_mean].values
        lo = ys - m_data[col_low].values
        hi = m_data[col_high].values - ys

        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            marker="o",
            markersize=7,
            capsize=4,
            label=_short_model_name(model),
            color=palette[idx],
            linewidth=1.5,
        )

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Faithfulness Score")
    ax.set_title("Quantization Effect on Faithfulness")
    ax.set_xticks(range(len(quant_order)))
    ax.set_xticklabels(quant_order)
    ax.set_ylim(0, 1.05)
    ax.legend(
        title="Model",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
    )

    out = output_dir / "quantization_effect.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_throughput_heatmap(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Chart 5: Throughput heatmap (model x hardware tier).

    Annotated heatmap showing tokens/second. Averaged across quants and
    datasets. NaN for infeasible combos.
    """
    col_mean = "tokens_per_second_mean"

    grouped = (
        df.groupby(["model", "hardware_tier"])
        .agg({col_mean: "mean"})
        .reset_index()
    )

    models = _sort_models_by_params(grouped["model"].unique().tolist())
    hw_tiers = sorted(grouped["hardware_tier"].unique().tolist())

    # Build pivot matrix
    pivot = pd.DataFrame(index=models, columns=hw_tiers, dtype=float)
    for _, row in grouped.iterrows():
        pivot.loc[row["model"], row["hardware_tier"]] = row[col_mean]

    # Rename index for display
    pivot.index = [_short_model_name(m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    sns.heatmap(
        pivot.astype(float),
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Tokens/Second"},
        ax=ax,
        mask=pivot.isna(),
    )
    ax.set_title("Throughput Heatmap: Model x Hardware Tier")
    ax.set_xlabel("Hardware Tier")
    ax.set_ylabel("Model")

    out = output_dir / "throughput_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_vram_usage(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Chart 6: VRAM usage bar chart by model and quantization.

    Horizontal bar chart showing peak VRAM (MB converted to GB for
    readability). Only GPU tiers (v100, 1660s).
    """
    col_mean = "vram_usage_mb_mean"
    col_low = "vram_usage_mb_ci_low"
    col_high = "vram_usage_mb_ci_high"

    # Filter to GPU hardware only
    gpu_df = df[df["hardware_tier"].isin(["v100", "mi25", "1660s"])].copy()
    if gpu_df.empty:
        gpu_df = df.copy()

    # Average across datasets, keep model x quant
    grouped = (
        gpu_df.groupby(["model", "quantization"])
        .agg({col_mean: "mean", col_low: "mean", col_high: "mean"})
        .reset_index()
    )
    grouped = grouped[grouped[col_mean] > 0]

    if grouped.empty:
        fig, ax = plt.subplots(figsize=FIGSIZE_VRAM)
        ax.set_title("VRAM Usage (No GPU Data)")
        out = output_dir / "vram_usage.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    # Convert to GB
    grouped["vram_gb"] = grouped[col_mean] / 1024
    grouped["err_low_gb"] = (grouped[col_mean] - grouped[col_low]) / 1024
    grouped["err_high_gb"] = (grouped[col_high] - grouped[col_mean]) / 1024

    # Create label
    grouped["label"] = grouped.apply(
        lambda r: f"{_short_model_name(r['model'])} ({r['quantization']})", axis=1
    )

    # Sort by VRAM usage
    grouped = grouped.sort_values("vram_gb", ascending=True)

    palette = sns.color_palette(PALETTE_NAME, len(grouped))

    fig, ax = plt.subplots(figsize=FIGSIZE_VRAM)
    y_pos = np.arange(len(grouped))

    ax.barh(
        y_pos,
        grouped["vram_gb"].values,
        xerr=[grouped["err_low_gb"].values, grouped["err_high_gb"].values],
        capsize=3,
        color=palette,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    # Add V100 32GB reference line
    ax.axvline(x=32, color="red", linestyle="--", linewidth=1, alpha=0.7, label="V100 32GB")
    # Add 1660S 6GB reference line
    ax.axvline(x=6, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="1660S 6GB")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(grouped["label"].values)
    ax.set_xlabel("Peak VRAM Usage (GB)")
    ax.set_title("VRAM Usage by Model and Quantization")
    ax.legend(loc="lower right")

    out = output_dir / "vram_usage.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

ALL_CHART_FUNCTIONS = [
    chart_faithfulness_by_model,
    chart_self_knowledge_by_model,
    chart_model_size_vs_faithfulness,
    chart_quantization_effect,
    chart_throughput_heatmap,
    chart_vram_usage,
]


def generate_all_charts(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Generate all publication-quality charts from an aggregated DataFrame.

    Args:
        df: DataFrame matching the v2 aggregated CSV schema.
        output_dir: Directory to save PNG files.

    Returns:
        List of paths to generated PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()

    generated: list[Path] = []
    for chart_fn in ALL_CHART_FUNCTIONS:
        try:
            path = chart_fn(df, output_dir)
            generated.append(path)
            print(f"  Generated: {path.name}")
        except Exception as e:
            print(f"  FAILED: {chart_fn.__name__}: {e}", file=sys.stderr)

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CompRAG Results Visualizer -- publication-quality charts from aggregated data",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/aggregated/summary.csv",
        help="Path to aggregated CSV file (default: results/aggregated/summary.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures/",
        help="Output directory for PNG charts (default: results/figures/)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate charts from synthetic demo data (no input file needed)",
    )

    args = parser.parse_args()

    if args.demo:
        print("Generating synthetic demo data...")
        df = generate_demo_data()
        print(f"  {len(df)} synthetic records generated")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading data from {input_path}...")
        df = load_aggregated_csv(input_path)
        print(f"  {len(df)} records loaded")

    print(f"Generating charts in {args.output_dir}...")
    generated = generate_all_charts(df, args.output_dir)

    print(f"\nDone. {len(generated)} charts generated:")
    for p in generated:
        print(f"  {p}")

    if len(generated) < 6:
        print(f"\nWARNING: Expected 6 charts, got {len(generated)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
