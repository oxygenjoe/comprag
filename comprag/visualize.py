"""Matplotlib figures for CompRAG evaluation results.

Reads aggregated JSONL, produces figures showing CU, SK, and Preference_Gap
vs quantization level with frontier reference lines.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

QUANT_ORDER: list[str] = ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16"]

MODEL_COLORS: dict[str, str] = {
    "qwen2.5-14b-instruct": "#1f77b4", "phi-4-14b": "#ff7f0e",
    "llama-3.1-8b-instruct": "#2ca02c", "qwen2.5-7b-instruct": "#d62728",
    "mistral-nemo-12b-instruct": "#9467bd", "gemma-2-9b-instruct": "#8c564b",
    "smollm2-1.7b-instruct": "#7f7f7f",
}

FRONTIER_STYLES: dict[str, str] = {
    "gpt-4o": "#e377c2", "gpt-5": "#bcbd22",
    "claude-3.5-sonnet": "#17becf", "gemini-2.0-flash": "#aec7e8",
}


def load_aggregated(input_dir: str) -> list[dict[str, Any]]:
    """Load all aggregated JSONL files from a directory."""
    path = Path(input_dir)
    records: list[dict[str, Any]] = []
    for jsonl_file in sorted(path.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info("Loaded %d aggregated records from %s", len(records), input_dir)
    return records


def _quant_x(quant: str) -> int | None:
    """Map quantization label to x-axis position. None if unknown."""
    return QUANT_ORDER.index(quant) if quant in QUANT_ORDER else None


def _filter_local(records: list[dict[str, Any]], pass_prefix: str) -> list[dict[str, Any]]:
    """Filter to local records with valid quant matching pass prefix."""
    return [r for r in records if r["pass"].startswith(pass_prefix)
            and _quant_x(r["quantization"]) is not None
            and r.get("source", "local") == "local"]


def _get_frontier(records: list[dict[str, Any]], pass_prefix: str) -> list[dict[str, Any]]:
    """Get frontier (API) model records for reference lines."""
    return [r for r in records if r["pass"].startswith(pass_prefix)
            and r.get("source") == "frontier"]


def _draw_frontier_lines(
    ax: plt.Axes, frontier: list[dict[str, Any]], metric_key: str,
) -> None:
    """Draw horizontal dashed reference lines with shaded CI bands."""
    seen: set[str] = set()
    for rec in frontier:
        model = rec["model"]
        if model in seen:
            continue
        seen.add(model)
        m = rec["metrics"][metric_key]
        color = FRONTIER_STYLES.get(model, "#999999")
        ax.axhline(m["mean"], linestyle="--", color=color, alpha=0.8, label=model)
        ax.axhspan(m["ci_lo"], m["ci_hi"], color=color, alpha=0.10)


def _build_series(
    records: list[dict[str, Any]], metric_key: str,
) -> dict[str, dict[str, list]]:
    """Group records by model into plottable series with x/y/ci/degraded."""
    series: dict[str, dict[str, list]] = {}
    for rec in records:
        model = rec["model"]
        if model not in series:
            series[model] = {"x": [], "y": [], "ci_lo": [], "ci_hi": [], "degraded": []}
        m = rec["metrics"][metric_key]
        s = series[model]
        s["x"].append(_quant_x(rec["quantization"]))
        s["y"].append(m["mean"])
        s["ci_lo"].append(m["ci_lo"])
        s["ci_hi"].append(m["ci_hi"])
        s["degraded"].append(rec.get("capability_degraded", False))
    return series


def _finish_axes(ax: plt.Axes, ylabel: str, title: str) -> None:
    """Apply shared axis formatting."""
    ax.set_xticks(range(len(QUANT_ORDER)))
    ax.set_xticklabels(QUANT_ORDER, rotation=30, ha="right")
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


def _save_fig(fig: plt.Figure, output_path: Path) -> Path:
    """Save figure, close, log, return path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)
    return output_path


def _plot_series_on_ax(
    ax: plt.Axes, series: dict[str, dict[str, list]], marker: str = "o",
) -> None:
    """Plot all model series with error bars and hollow degraded markers."""
    for model, data in sorted(series.items()):
        color = MODEL_COLORS.get(model, "#333333")
        x, y = np.array(data["x"]), np.array(data["y"])
        ci_lo, ci_hi = np.array(data["ci_lo"]), np.array(data["ci_hi"])
        degraded = np.array(data["degraded"])
        order = np.argsort(x)
        x, y, ci_lo, ci_hi, degraded = (
            x[order], y[order], ci_lo[order], ci_hi[order], degraded[order])
        yerr = np.array([y - ci_lo, ci_hi - y])
        ax.errorbar(x, y, yerr=yerr, color=color, marker=marker,
                     label=model, capsize=3, linewidth=1.5)
        if degraded.any():
            ax.scatter(x[degraded], y[degraded], facecolors="none",
                       edgecolors=color, s=100, zorder=5, linewidths=2)


def _plot_metric_vs_quant(
    records: list[dict[str, Any]], metric_key: str,
    ylabel: str, title: str, output_path: Path,
) -> Path:
    """Shared logic: metric vs quantization with frontier references."""
    local = _filter_local(records, "pass3")
    frontier = _get_frontier(records, "pass3")
    series = _build_series(local, metric_key)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_series_on_ax(ax, series)
    _draw_frontier_lines(ax, frontier, metric_key)
    _finish_axes(ax, ylabel, title)
    return _save_fig(fig, output_path)


def plot_cu_vs_quant(records: list[dict[str, Any]], output_dir: str) -> Path:
    """CU vs Quantization: one curve per model, frontier dashed lines."""
    return _plot_metric_vs_quant(
        records, "cu", "Context Utilization (CU)",
        "Context Utilization vs Quantization Level",
        Path(output_dir) / "cu_vs_quant.png")


def plot_sk_vs_quant(records: list[dict[str, Any]], output_dir: str) -> Path:
    """SK vs Quantization: one curve per model, frontier dashed lines."""
    return _plot_metric_vs_quant(
        records, "sk", "Self Knowledge (SK)",
        "Self Knowledge vs Quantization Level",
        Path(output_dir) / "sk_vs_quant.png")


def plot_preference_gap_vs_quant(
    records: list[dict[str, Any]], output_dir: str,
) -> Path:
    """Preference Gap vs Quantization: Pass3_CU - Pass2_CU per model."""
    local = _filter_local(records, "pass3")
    series = _build_series(local, "preference_gap")
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_series_on_ax(ax, series, marker="s")
    ax.axhline(0, linestyle=":", color="black", alpha=0.5)
    _finish_axes(ax, "Preference Gap (Pass3 CU - Pass2 CU)",
                 "Preference Gap vs Quantization Level")
    return _save_fig(fig, Path(output_dir) / "preference_gap_vs_quant.png")
