#!/usr/bin/env python3
"""Download GGUF model files from HuggingFace per models.yaml registry.

Supports downloading individual models by name/quant, by priority tier,
or all models at once. Uses huggingface_hub for downloads with built-in
progress display. Idempotent: skips already-downloaded files that pass
integrity checks.

Usage:
    # Single model + quant
    python scripts/download_models.py --model llama-3.1-8b-instruct --quant Q4_K_M

    # All quants for a model
    python scripts/download_models.py --model llama-3.1-8b-instruct

    # All HIGH priority models
    python scripts/download_models.py --priority HIGH

    # Everything in the registry
    python scripts/download_models.py --all

    # Dry run (show what would be downloaded)
    python scripts/download_models.py --all --dry-run

    # List available models
    python scripts/download_models.py --list
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is importable when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from comprag.utils import get_logger, load_config, Timer

logger = get_logger("comprag.download_models")

# Default models directory
MODELS_DIR = _PROJECT_ROOT / "models"


def load_model_registry(config_dir: Optional[Path] = None) -> dict:
    """Load and return the models section from models.yaml.

    Args:
        config_dir: Override config directory.

    Returns:
        Dict of model_id -> model_spec from models.yaml.

    Raises:
        FileNotFoundError: If models.yaml is missing.
        ValueError: If models.yaml has no 'models' key.
    """
    cfg = load_config("models", config_dir=config_dir)
    models = cfg.get("models")
    if not models:
        raise ValueError("models.yaml has no 'models' key or it is empty")
    return models


def normalize_priority(priority_str: str) -> str:
    """Normalize priority strings for comparison.

    Handles values like 'HIGH', 'HIGH (baseline)', 'LOW (floor test)',
    'LOW (FPGA only)' by extracting the first word.

    Args:
        priority_str: Raw priority string from models.yaml.

    Returns:
        Uppercase priority tier: HIGH, MEDIUM, or LOW.
    """
    return str(priority_str).strip().split()[0].upper()


def filter_models(
    registry: dict,
    model_id: Optional[str] = None,
    quant: Optional[str] = None,
    priority: Optional[str] = None,
    download_all: bool = False,
) -> list[tuple[str, str, dict, dict]]:
    """Filter the model registry into a list of download targets.

    Args:
        registry: Full model registry from models.yaml.
        model_id: Specific model ID to download (e.g. 'llama-3.1-8b-instruct').
        quant: Specific quantization to download (e.g. 'Q4_K_M').
        priority: Priority tier filter (e.g. 'HIGH').
        download_all: If True, include all models.

    Returns:
        List of (model_id, quant_id, quant_spec, model_spec) tuples.

    Raises:
        ValueError: If model_id or quant not found in registry.
    """
    targets = []

    if model_id:
        if model_id not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Model '{model_id}' not found in registry. "
                f"Available: {available}"
            )
        model_spec = registry[model_id]
        quants = model_spec.get("quantizations", {})

        if quant:
            if quant not in quants:
                available_quants = ", ".join(sorted(quants.keys()))
                raise ValueError(
                    f"Quantization '{quant}' not available for '{model_id}'. "
                    f"Available: {available_quants}"
                )
            targets.append((model_id, quant, quants[quant], model_spec))
        else:
            # All quants for this model
            for q_id, q_spec in quants.items():
                targets.append((model_id, q_id, q_spec, model_spec))

    elif priority:
        priority_upper = priority.strip().upper()
        for mid, mspec in registry.items():
            if normalize_priority(mspec.get("priority", "")) == priority_upper:
                for q_id, q_spec in mspec.get("quantizations", {}).items():
                    targets.append((mid, q_id, q_spec, mspec))

        if not targets:
            raise ValueError(
                f"No models found with priority '{priority}'. "
                f"Valid tiers: HIGH, MEDIUM, LOW"
            )

    elif download_all:
        for mid, mspec in registry.items():
            for q_id, q_spec in mspec.get("quantizations", {}).items():
                targets.append((mid, q_id, q_spec, mspec))
    else:
        raise ValueError(
            "Must specify --model, --priority, or --all. "
            "Use --list to see available models."
        )

    return targets


def is_already_downloaded(filepath: Path, expected_size_gb: float) -> bool:
    """Check if a model file already exists and passes integrity check.

    Verifies the file exists and its size is within 5% of the expected size.
    This catches partial downloads and corruption without requiring checksums
    (which HuggingFace GGUF repos don't consistently provide).

    Args:
        filepath: Path to the local GGUF file.
        expected_size_gb: Expected file size in GB from models.yaml.

    Returns:
        True if the file exists and passes size check.
    """
    if not filepath.exists():
        return False

    actual_size_bytes = filepath.stat().st_size
    expected_size_bytes = expected_size_gb * (1024 ** 3)

    # Allow 5% tolerance (GGUF sizes in yaml are approximate)
    lower = expected_size_bytes * 0.95
    upper = expected_size_bytes * 1.05

    if lower <= actual_size_bytes <= upper:
        return True

    # File exists but wrong size — likely partial download
    actual_gb = actual_size_bytes / (1024 ** 3)
    logger.warning(
        "File %s exists but size mismatch: %.2f GB actual vs %.2f GB expected. "
        "Will re-download.",
        filepath.name,
        actual_gb,
        expected_size_gb,
    )
    return False


def download_model(
    model_id: str,
    quant_id: str,
    quant_spec: dict,
    model_spec: dict,
    models_dir: Path = MODELS_DIR,
    dry_run: bool = False,
    force: bool = False,
    hf_token: Optional[str] = None,
) -> bool:
    """Download a single GGUF model file from HuggingFace.

    Args:
        model_id: Model identifier (e.g. 'llama-3.1-8b-instruct').
        quant_id: Quantization format (e.g. 'Q4_K_M').
        quant_spec: Quant specification dict with 'filename' and 'size_gb'.
        model_spec: Full model specification dict with 'hf_repo', etc.
        models_dir: Local directory to store downloaded files.
        dry_run: If True, only print what would be downloaded.
        force: If True, re-download even if file exists and passes checks.
        hf_token: Optional HuggingFace API token for gated models.

    Returns:
        True if download succeeded or file already exists, False on error.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        GatedRepoError,
    )

    hf_repo = model_spec.get("hf_repo", "")
    filename = quant_spec.get("filename", "")
    size_gb = quant_spec.get("size_gb", 0)
    display_name = model_spec.get("display_name", model_id)

    if not hf_repo or hf_repo == "TBD":
        logger.warning(
            "Skipping %s %s: HuggingFace repo is TBD",
            display_name,
            quant_id,
        )
        return False

    if not filename or filename == "TBD":
        logger.warning(
            "Skipping %s %s: filename is TBD",
            display_name,
            quant_id,
        )
        return False

    local_path = models_dir / filename

    if dry_run:
        status = "EXISTS" if local_path.exists() else "DOWNLOAD"
        logger.info(
            "[DRY RUN] %s | %s %s | %s | %.1f GB",
            status,
            display_name,
            quant_id,
            filename,
            size_gb,
        )
        return True

    # Check if already downloaded (skip unless --force)
    if not force and is_already_downloaded(local_path, size_gb):
        logger.info(
            "SKIP %s %s — already downloaded (%.1f GB): %s",
            display_name,
            quant_id,
            size_gb,
            local_path,
        )
        return True

    # Ensure models directory exists
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading %s %s (%.1f GB) from %s ...",
        display_name,
        quant_id,
        size_gb,
        hf_repo,
    )

    try:
        with Timer() as t:
            downloaded_path = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False,
                token=hf_token,
            )

        downloaded_path = Path(downloaded_path)

        # Verify downloaded file integrity
        if not downloaded_path.exists():
            logger.error(
                "FAIL %s %s: download completed but file not found at %s",
                display_name,
                quant_id,
                downloaded_path,
            )
            return False

        actual_gb = downloaded_path.stat().st_size / (1024 ** 3)

        # Size sanity check (warn but don't fail — yaml sizes are approximate)
        if size_gb > 0:
            ratio = actual_gb / size_gb
            if ratio < 0.90 or ratio > 1.10:
                logger.warning(
                    "Size mismatch for %s: expected ~%.1f GB, got %.2f GB (ratio: %.2f). "
                    "File may be corrupt or yaml size is inaccurate.",
                    filename,
                    size_gb,
                    actual_gb,
                    ratio,
                )

        speed_mbps = (actual_gb * 1024) / t.elapsed if t.elapsed > 0 else 0
        logger.info(
            "OK %s %s — %.2f GB in %.1fs (%.1f MB/s): %s",
            display_name,
            quant_id,
            actual_gb,
            t.elapsed,
            speed_mbps,
            downloaded_path,
        )
        return True

    except RepositoryNotFoundError:
        logger.error(
            "FAIL %s %s: repository '%s' not found on HuggingFace",
            display_name,
            quant_id,
            hf_repo,
        )
        return False

    except EntryNotFoundError:
        logger.error(
            "FAIL %s %s: file '%s' not found in repo '%s'",
            display_name,
            quant_id,
            filename,
            hf_repo,
        )
        return False

    except GatedRepoError:
        logger.error(
            "FAIL %s %s: repo '%s' is gated. Set HF_TOKEN env var or use --token.",
            display_name,
            quant_id,
            hf_repo,
        )
        return False

    except Exception as e:
        logger.error(
            "FAIL %s %s: unexpected error: %s: %s",
            display_name,
            quant_id,
            type(e).__name__,
            e,
        )
        return False


def download_batch(
    targets: list[tuple[str, str, dict, dict]],
    models_dir: Path = MODELS_DIR,
    dry_run: bool = False,
    force: bool = False,
    hf_token: Optional[str] = None,
) -> tuple[int, int, int]:
    """Download a batch of model files.

    Processes each target sequentially. Errors on individual downloads
    do not halt the batch.

    Args:
        targets: List of (model_id, quant_id, quant_spec, model_spec) tuples.
        models_dir: Local directory to store downloaded files.
        dry_run: If True, only print what would be downloaded.
        force: If True, re-download even if files exist.
        hf_token: Optional HuggingFace API token.

    Returns:
        Tuple of (success_count, skip_count, fail_count).
    """
    total = len(targets)
    total_size_gb = sum(t[2].get("size_gb", 0) for t in targets)

    logger.info(
        "Download plan: %d file(s), ~%.1f GB total",
        total,
        total_size_gb,
    )

    successes = 0
    failures = 0

    for i, (model_id, quant_id, quant_spec, model_spec) in enumerate(targets, 1):
        logger.info("--- [%d/%d] ---", i, total)
        ok = download_model(
            model_id=model_id,
            quant_id=quant_id,
            quant_spec=quant_spec,
            model_spec=model_spec,
            models_dir=models_dir,
            dry_run=dry_run,
            force=force,
            hf_token=hf_token,
        )
        if ok:
            successes += 1
        else:
            failures += 1

    logger.info(
        "Done: %d succeeded, %d failed out of %d total",
        successes,
        failures,
        total,
    )

    return successes, failures, total


def list_models(registry: dict) -> None:
    """Print a formatted table of all models in the registry.

    Args:
        registry: Model registry dict from models.yaml.
    """
    print(f"\n{'Model ID':<35} {'Display Name':<30} {'Priority':<15} {'Quants'}")
    print("-" * 110)

    for model_id, spec in registry.items():
        display = spec.get("display_name", model_id)
        priority = str(spec.get("priority", "?"))
        quants = spec.get("quantizations", {})
        quant_strs = []
        for q_id, q_spec in quants.items():
            size = q_spec.get("size_gb", "?")
            quant_strs.append(f"{q_id} ({size}GB)")
        quant_summary = ", ".join(quant_strs)
        print(f"{model_id:<35} {display:<30} {priority:<15} {quant_summary}")

    total_files = sum(
        len(spec.get("quantizations", {})) for spec in registry.values()
    )
    total_gb = sum(
        q.get("size_gb", 0)
        for spec in registry.values()
        for q in spec.get("quantizations", {}).values()
    )
    print(f"\nTotal: {len(registry)} models, {total_files} files, ~{total_gb:.1f} GB")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the download script.

    Returns:
        Configured argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Download GGUF model files from HuggingFace per models.yaml registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --model llama-3.1-8b-instruct --quant Q4_K_M
  %(prog)s --model llama-3.1-8b-instruct
  %(prog)s --priority HIGH
  %(prog)s --all
  %(prog)s --all --dry-run
  %(prog)s --list
""",
    )

    # Selection group (mutually exclusive)
    select = parser.add_mutually_exclusive_group()
    select.add_argument(
        "--model",
        type=str,
        metavar="ID",
        help="Model ID to download (e.g. 'llama-3.1-8b-instruct'). "
        "Downloads all quants unless --quant is specified.",
    )
    select.add_argument(
        "--priority",
        type=str,
        metavar="TIER",
        choices=["HIGH", "MEDIUM", "LOW", "high", "medium", "low"],
        help="Download all models of a priority tier (HIGH, MEDIUM, LOW).",
    )
    select.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="Download all models in the registry.",
    )
    select.add_argument(
        "--list",
        action="store_true",
        dest="list_models",
        help="List all available models and exit.",
    )

    parser.add_argument(
        "--quant",
        type=str,
        metavar="FORMAT",
        help="Specific quantization format (e.g. 'Q4_K_M'). Only with --model.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        metavar="DIR",
        help=f"Directory to store downloaded model files (default: {MODELS_DIR}).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override config directory (default: <project>/config/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist and pass integrity checks.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        metavar="HF_TOKEN",
        help="HuggingFace API token for gated repos. "
        "Also reads from HF_TOKEN environment variable.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the download script.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on partial failure, 2 on fatal error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate --quant requires --model
    if args.quant and not args.model:
        parser.error("--quant requires --model")

    # Resolve HF token
    hf_token = args.token or os.environ.get("HF_TOKEN")

    try:
        registry = load_model_registry(config_dir=args.config_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load model registry: %s", e)
        return 2

    # List mode
    if args.list_models:
        list_models(registry)
        return 0

    # Need at least one selection flag
    if not args.model and not args.priority and not args.download_all:
        parser.error("Must specify --model, --priority, --all, or --list")

    try:
        targets = filter_models(
            registry=registry,
            model_id=args.model,
            quant=args.quant,
            priority=args.priority,
            download_all=args.download_all,
        )
    except ValueError as e:
        logger.error("%s", e)
        return 2

    if not targets:
        logger.warning("No download targets matched. Nothing to do.")
        return 0

    successes, failures, total = download_batch(
        targets=targets,
        models_dir=args.models_dir,
        dry_run=args.dry_run,
        force=args.force,
        hf_token=hf_token,
    )

    if failures > 0:
        logger.error(
            "%d/%d downloads failed. Re-run to retry failed downloads.",
            failures,
            total,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
