#!/usr/bin/env python3
"""Download GGUF model files from HuggingFace for CompRAG evaluation.

Reads the model registry from config/models.yaml and downloads GGUF files.
Supports resuming interrupted downloads and skipping complete files.
"""

import argparse
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
HF_BASE_URL = "https://huggingface.co"


def load_model_registry(config_path: Path) -> dict[str, Any]:
    """Load model definitions from models.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "models" not in data:
        raise ValueError(f"Invalid model config: {config_path}")
    return data["models"]


def list_available_models(registry: dict[str, Any]) -> None:
    """Print available models and their quants."""
    print("Available models:\n")
    for key, info in registry.items():
        quants = ", ".join(info.get("quants", {}).keys())
        print(f"  {key}")
        print(f"    Name: {info.get('display_name', key)}  Role: {info.get('role', '?')}")
        print(f"    Repo: {info.get('hf_repo', 'N/A')}  Quants: {quants}\n")


def get_remote_file_size(url: str) -> Optional[int]:
    """Get file size from server via HEAD request."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return None


def file_already_complete(filepath: Path, expected_size: Optional[int]) -> bool:
    """Check if file exists and matches expected size."""
    if not filepath.exists():
        return False
    local_size = filepath.stat().st_size
    if expected_size is None:
        logger.info("File exists, no remote size — skipping %s", filepath.name)
        return True
    if local_size == expected_size:
        logger.info("Already complete (%s, %.2f GB) — skipping", filepath.name, local_size / 1e9)
        return True
    if local_size < expected_size:
        logger.info("Partial %s: %.2f/%.2f GB — will resume", filepath.name, local_size / 1e9, expected_size / 1e9)
        return False
    logger.warning("Local %s larger than remote — re-downloading", filepath.name)
    filepath.unlink()
    return False


def _do_download(url: str, filepath: Path) -> None:
    """Execute a single download attempt with resume support."""
    existing_size = filepath.stat().st_size if filepath.exists() else 0
    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")

    with urllib.request.urlopen(req, timeout=60) as resp:
        if existing_size > 0 and resp.status != 206:
            logger.warning("Server does not support resume — restarting")
            existing_size = 0
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) + existing_size if total else None
        mode = "ab" if existing_size > 0 and resp.status == 206 else "wb"
        downloaded = existing_size if mode == "ab" else 0
        last_pct = -1

        with open(filepath, mode) as f:
            while chunk := resp.read(8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = int((downloaded / total_bytes) * 100) // 10 * 10
                    if pct > last_pct:
                        logger.info("  %s: %d%% (%.2f GB)", filepath.name, pct, downloaded / 1e9)
                        last_pct = pct

    logger.info("Download complete: %s (%.2f GB)", filepath.name, filepath.stat().st_size / 1e9)


def download_file_with_resume(url: str, filepath: Path, max_retries: int = 3) -> None:
    """Download a file with resume support and retry logic."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            _do_download(url, filepath)
            return
        except (urllib.error.URLError, OSError) as e:
            if attempt == max_retries:
                logger.error("Failed after %d attempts: %s — %s", max_retries, filepath.name, e)
                raise
            wait = 2 ** attempt
            logger.warning("Attempt %d/%d failed for %s — retrying in %ds", attempt, max_retries, filepath.name, wait)
            time.sleep(wait)


def download_model_quant(model_key: str, model_info: dict[str, Any], quant: str, models_dir: Path) -> bool:
    """Download a single model quant. Returns True on success."""
    quants = model_info.get("quants", {})
    if quant not in quants:
        logger.error("Quant '%s' not in '%s'. Available: %s", quant, model_key, ", ".join(quants.keys()))
        return False
    filename = quants[quant]
    url = f"{HF_BASE_URL}/{model_info['hf_repo']}/resolve/main/{filename}"
    filepath = models_dir / filename
    if file_already_complete(filepath, get_remote_file_size(url)):
        return True
    logger.info("Downloading %s/%s -> %s", model_key, quant, filepath)
    try:
        download_file_with_resume(url, filepath)
        return True
    except Exception as e:
        logger.error("Failed %s/%s: %s", model_key, quant, e)
        return False


def download_model_all_quants(model_key: str, model_info: dict[str, Any], models_dir: Path) -> tuple[int, int]:
    """Download all quants for a model. Returns (ok, fail) counts."""
    ok, fail = 0, 0
    for quant in model_info.get("quants", {}):
        if download_model_quant(model_key, model_info, quant, models_dir):
            ok += 1
        else:
            fail += 1
    return ok, fail


def download_all_models(registry: dict[str, Any], models_dir: Path) -> tuple[int, int]:
    """Download all models and quants. Returns (ok, fail) counts."""
    total_ok, total_fail = 0, 0
    for model_key, model_info in registry.items():
        logger.info("--- Processing model: %s ---", model_key)
        ok, fail = download_model_all_quants(model_key, model_info, models_dir)
        total_ok += ok
        total_fail += fail
    return total_ok, total_fail


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Download GGUF model files from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  %(prog)s --all\n  %(prog)s --model qwen2.5-14b-instruct --quant Q4_K_M\n  %(prog)s --list",
    )
    p.add_argument("--model", type=str, default=None, help="Model key from models.yaml")
    p.add_argument("--quant", type=str, default=None, help="Specific quant (e.g. Q4_K_M). All if omitted")
    p.add_argument("--all", action="store_true", dest="download_all", help="Download all models and quants")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Model save directory")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to models.yaml")
    p.add_argument("--list", action="store_true", dest="list_models", help="List available models and exit")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the model downloader."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        registry = load_model_registry(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load model config: %s", e)
        return 1

    if args.list_models:
        list_available_models(registry)
        return 0
    if not args.download_all and not args.model:
        logger.error("Specify --model <name>, --all, or --list")
        return 1

    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Models directory: %s", models_dir)

    if args.download_all:
        ok, fail = download_all_models(registry, models_dir)
        logger.info("Complete: %d succeeded, %d failed", ok, fail)
        return 1 if fail > 0 else 0

    if args.model not in registry:
        logger.error("Unknown model '%s'. Available: %s", args.model, ", ".join(registry.keys()))
        return 1
    model_info = registry[args.model]
    if args.quant:
        return 0 if download_model_quant(args.model, model_info, args.quant, models_dir) else 1
    ok, fail = download_model_all_quants(args.model, model_info, models_dir)
    logger.info("Complete: %d succeeded, %d failed", ok, fail)
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
