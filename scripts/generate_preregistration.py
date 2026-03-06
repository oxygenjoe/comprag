#!/usr/bin/env python3
"""Generate PRE_REGISTRATION.md with artifact hashes, tool versions, and study design.

Collects GGUF SHA256s, RAGChecker version, RGB download date, hypothesis text,
predictions, and statistical criteria. Must be committed before any experimental run.
"""

import argparse
import hashlib
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

HYPOTHESIS = (
    "Local small models (8-14B parameters) with retrieval-augmented generation "
    "produce higher context utilization (CU) scores than the same models without "
    "retrieval, and approach frontier model CU at Q8_0 and FP16 quantization "
    "levels on factoid QA tasks."
)

PREDICTIONS = [
    "P1: For each primary-architecture model, CU increases monotonically from "
    "Q3_K_M to FP16 on RGB counterfactual subset (pass2_loose).",
    "P2: At Q8_0 and above, primary-architecture models achieve CU within 0.15 "
    "of the best frontier model on the same dataset/subset/pass.",
    "P3: Preference_Gap (pass3_CU - pass2_CU) is negative or near-zero for "
    "primary-architecture models at Q4_K_M and above, indicating strict prompts "
    "do not improve over loose prompts when retrieval quality is held constant.",
    "P4: SmolLM2 1.7B (floor model) achieves CU < 0.30 on pass2_loose, "
    "confirming that sub-2B models lack sufficient capacity for RAG.",
    "P5: Secondary-architecture models (Mistral NeMo, Gemma 2) at Q4_K_M "
    "achieve CU within 0.10 of the matched primary-architecture model at Q4_K_M.",
]

STATISTICAL_CRITERIA = {
    "bootstrap_resamples": 1000,
    "confidence_level": 0.95,
    "capability_degradation_threshold": 0.30,
    "significance": "Non-overlapping 95% bootstrap CIs constitute significant difference.",
    "effect_size": "CU difference > 0.05 with non-overlapping CIs is practically significant.",
}


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def collect_gguf_hashes(models_dir: Path) -> list[dict[str, str]]:
    """Collect SHA256 hashes for all .gguf files in models_dir."""
    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        logger.warning("No .gguf files found in %s", models_dir)
        return []

    results = []
    for gf in gguf_files:
        logger.info("Hashing %s ...", gf.name)
        results.append({"filename": gf.name, "sha256": sha256_file(gf)})
    return results


def get_ragchecker_version() -> str:
    """Get installed RAGChecker version, or 'not installed'."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "ragchecker"],
            capture_output=True, text=True, check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        logger.warning("Could not determine RAGChecker version")
    return "not installed"


def get_rgb_download_date(datasets_dir: Path) -> str:
    """Get RGB dataset download date from directory mtime."""
    rgb_dir = datasets_dir / "rgb"
    if rgb_dir.exists():
        mtime = rgb_dir.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    return "not downloaded"


def load_model_config(config_path: Path) -> dict:
    """Load models.yaml and return the models dict."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


def format_preregistration(
    gguf_hashes: list[dict[str, str]],
    ragchecker_version: str,
    rgb_date: str,
    model_config: dict,
) -> str:
    """Format the full PRE_REGISTRATION.md content."""
    lines: list[str] = []
    lines.append("# Pre-Registration: CompRAG Quantization Benchmark")
    lines.append("")
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"Generated: {ts}")
    lines.append("")

    _append_hypothesis(lines)
    _append_predictions(lines)
    _append_statistical_criteria(lines)
    _append_models(lines, model_config)
    _append_artifact_hashes(lines, gguf_hashes)
    _append_tool_versions(lines, ragchecker_version, rgb_date)

    lines.append("---")
    lines.append("")
    lines.append("This document was generated before any experimental runs.")
    lines.append("Commit this file to lock the experimental design.")
    lines.append("")
    return "\n".join(lines)


def _append_hypothesis(lines: list[str]) -> None:
    lines.append("## Hypothesis")
    lines.append("")
    lines.append(HYPOTHESIS)
    lines.append("")


def _append_predictions(lines: list[str]) -> None:
    lines.append("## Predictions")
    lines.append("")
    for p in PREDICTIONS:
        lines.append(f"- {p}")
    lines.append("")


def _append_statistical_criteria(lines: list[str]) -> None:
    lines.append("## Statistical Criteria")
    lines.append("")
    for key, val in STATISTICAL_CRITERIA.items():
        lines.append(f"- **{key}**: {val}")
    lines.append("")


def _append_models(lines: list[str], model_config: dict) -> None:
    lines.append("## Models")
    lines.append("")
    lines.append("| Model | Params | Role | Quants |")
    lines.append("|-------|--------|------|--------|")
    for name, cfg in model_config.items():
        quants = ", ".join(cfg.get("quants", {}).keys())
        lines.append(
            f"| {cfg.get('display_name', name)} "
            f"| {cfg.get('params', '?')} "
            f"| {cfg.get('role', '?')} "
            f"| {quants} |"
        )
    lines.append("")


def _append_artifact_hashes(
    lines: list[str], gguf_hashes: list[dict[str, str]]
) -> None:
    lines.append("## GGUF Artifact Hashes (SHA256)")
    lines.append("")
    if gguf_hashes:
        lines.append("| Filename | SHA256 |")
        lines.append("|----------|--------|")
        for entry in gguf_hashes:
            lines.append(f"| {entry['filename']} | `{entry['sha256']}` |")
    else:
        lines.append("*No GGUF files found. Hashes will be recorded before run.*")
    lines.append("")


def _append_tool_versions(
    lines: list[str], ragchecker_version: str, rgb_date: str
) -> None:
    lines.append("## Tool Versions and Data Provenance")
    lines.append("")
    lines.append(f"- **RAGChecker version**: {ragchecker_version}")
    lines.append(f"- **RGB download date**: {rgb_date}")
    lines.append(f"- **Python**: {sys.version.split()[0]}")
    lines.append("")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate PRE_REGISTRATION.md with artifact hashes and study design."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing .gguf model files (default: models/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PRE_REGISTRATION.md"),
        help="Output path for pre-registration document (default: PRE_REGISTRATION.md)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/models.yaml"),
        help="Path to models.yaml config (default: config/models.yaml)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Directory containing downloaded datasets (default: datasets/)",
    )
    parser.add_argument(
        "--skip-hashing",
        action="store_true",
        help="Skip GGUF SHA256 computation (useful for quick runs)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    if args.config.exists():
        model_config = load_model_config(args.config)
        logger.info("Loaded %d models from %s", len(model_config), args.config)
    else:
        logger.warning("Config not found at %s, using empty model list", args.config)
        model_config = {}

    if args.skip_hashing:
        logger.info("Skipping GGUF hashing (--skip-hashing)")
        gguf_hashes: list[dict[str, str]] = []
    else:
        gguf_hashes = collect_gguf_hashes(args.models_dir)

    ragchecker_version = get_ragchecker_version()
    rgb_date = get_rgb_download_date(args.datasets_dir)

    content = format_preregistration(
        gguf_hashes, ragchecker_version, rgb_date, model_config
    )

    args.output.write_text(content)
    logger.info("Wrote %s (%d bytes)", args.output, len(content))


if __name__ == "__main__":
    main()
