#!/usr/bin/env python3
"""Download evaluation datasets for the CUMRAG benchmark harness.

Downloads three datasets into the datasets/ directory:
  - RGB (Retrieval-Augmented Generation Benchmark) via git clone
  - Natural Questions (nq_open) via HuggingFace datasets
  - HaluEval (Hallucination Evaluation) via git clone

Runnable standalone (CLI with argparse) AND importable as module.
Idempotent: skips datasets that already exist.
Parallel: downloads independent datasets concurrently via ThreadPoolExecutor.
Fail loudly: network failures raise immediately with full context.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure project root is importable when run standalone
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cumrag.utils import get_logger, setup_logging, timer

logger = get_logger("cumrag.download_datasets")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASETS_DIR = _PROJECT_ROOT / "datasets"

RGB_REPO_URL = "https://github.com/chen700564/RGB.git"
HALUEVAL_REPO_URL = "https://github.com/RUCAIBox/HaluEval.git"
NQ_DATASET_NAME = "nq_open"


# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------


def _git_clone(url: str, dest: Path, shallow: bool = True) -> None:
    """Clone a git repository to dest. Raises on failure.

    Args:
        url: Git remote URL.
        dest: Local directory to clone into.
        shallow: If True, use --depth 1 for faster clone.
    """
    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    cmd.extend([url, str(dest)])

    logger.info("Cloning %s -> %s", url, dest)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min timeout for large repos
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed for {url}:\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    logger.info("Cloned %s successfully", url)


def _is_populated(directory: Path) -> bool:
    """Check if a directory exists and contains non-hidden files or subdirs."""
    if not directory.is_dir():
        return False
    # Check for any non-hidden content
    for item in directory.iterdir():
        if not item.name.startswith("."):
            return True
    return False


def download_rgb(datasets_dir: Path, force: bool = False) -> Path:
    """Download the RGB benchmark dataset via git clone.

    Args:
        datasets_dir: Parent directory for all datasets.
        force: If True, re-download even if already present.

    Returns:
        Path to the RGB dataset directory.

    Raises:
        RuntimeError: On clone failure.
    """
    dest = datasets_dir / "rgb"

    if _is_populated(dest) and not force:
        logger.info("RGB already downloaded at %s, skipping", dest)
        return dest

    # Clean up partial downloads
    if dest.exists() and force:
        logger.info("Removing existing RGB directory for re-download")
        shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

    # Clone into a temp dir then move contents, since dest may already exist
    tmp_clone = datasets_dir / "_rgb_clone_tmp"
    if tmp_clone.exists():
        shutil.rmtree(tmp_clone)

    try:
        _git_clone(RGB_REPO_URL, tmp_clone)
        # Move contents from tmp into dest
        dest.mkdir(parents=True, exist_ok=True)
        for item in tmp_clone.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
    finally:
        if tmp_clone.exists():
            shutil.rmtree(tmp_clone)

    logger.info("RGB dataset ready at %s", dest)
    return dest


def download_nq(datasets_dir: Path, force: bool = False) -> Path:
    """Download Natural Questions (nq_open) via HuggingFace datasets.

    Saves train and validation splits as JSONL files for consistency
    with the rest of the pipeline.

    Args:
        datasets_dir: Parent directory for all datasets.
        force: If True, re-download even if already present.

    Returns:
        Path to the NQ dataset directory.

    Raises:
        ImportError: If datasets library is not installed.
        RuntimeError: On download failure.
    """
    dest = datasets_dir / "nq"
    marker = dest / "_download_complete"

    if marker.exists() and not force:
        logger.info("NQ already downloaded at %s, skipping", dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for NQ download. "
            "Install it with: pip install datasets"
        )

    logger.info("Downloading nq_open from HuggingFace...")

    try:
        ds = load_dataset(NQ_DATASET_NAME, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {NQ_DATASET_NAME} from HuggingFace: {e}"
        ) from e

    # Save each split as JSONL
    for split_name in ds:
        split = ds[split_name]
        output_file = dest / f"{split_name}.jsonl"
        logger.info(
            "Writing NQ %s split (%d examples) to %s",
            split_name,
            len(split),
            output_file,
        )

        with open(output_file, "w", encoding="utf-8") as f:
            for row in split:
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                f.write(line + "\n")

    # Write completion marker
    marker.write_text("done\n")
    logger.info("NQ dataset ready at %s", dest)
    return dest


def download_halueval(datasets_dir: Path, force: bool = False) -> Path:
    """Download the HaluEval dataset via git clone.

    Args:
        datasets_dir: Parent directory for all datasets.
        force: If True, re-download even if already present.

    Returns:
        Path to the HaluEval dataset directory.

    Raises:
        RuntimeError: On clone failure.
    """
    dest = datasets_dir / "halueval"

    if _is_populated(dest) and not force:
        logger.info("HaluEval already downloaded at %s, skipping", dest)
        return dest

    # Clean up partial downloads
    if dest.exists() and force:
        logger.info("Removing existing HaluEval directory for re-download")
        shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

    # Clone into a temp dir then move contents
    tmp_clone = datasets_dir / "_halueval_clone_tmp"
    if tmp_clone.exists():
        shutil.rmtree(tmp_clone)

    try:
        _git_clone(HALUEVAL_REPO_URL, tmp_clone)
        dest.mkdir(parents=True, exist_ok=True)
        for item in tmp_clone.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
    finally:
        if tmp_clone.exists():
            shutil.rmtree(tmp_clone)

    logger.info("HaluEval dataset ready at %s", dest)
    return dest


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "rgb": download_rgb,
    "nq": download_nq,
    "halueval": download_halueval,
}


def download_all(
    datasets_dir: Optional[Path] = None,
    datasets: Optional[list[str]] = None,
    force: bool = False,
    parallel: bool = True,
) -> dict[str, Path]:
    """Download all (or selected) evaluation datasets.

    Args:
        datasets_dir: Directory to download into (default: <project>/datasets/).
        datasets: List of dataset names to download. None means all.
        force: Re-download even if already present.
        parallel: Download datasets concurrently (default True).

    Returns:
        Dict mapping dataset name -> local path.

    Raises:
        RuntimeError: If any download fails (fail loudly per spec).
    """
    if datasets_dir is None:
        datasets_dir = DEFAULT_DATASETS_DIR

    datasets_dir = Path(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to download
    names = datasets if datasets else list(DOWNLOADERS.keys())
    invalid = set(names) - set(DOWNLOADERS.keys())
    if invalid:
        raise ValueError(
            f"Unknown dataset(s): {invalid}. Valid: {list(DOWNLOADERS.keys())}"
        )

    results: dict[str, Path] = {}
    errors: dict[str, Exception] = {}

    if parallel and len(names) > 1:
        with ThreadPoolExecutor(max_workers=len(names)) as executor:
            futures = {
                executor.submit(DOWNLOADERS[name], datasets_dir, force): name
                for name in names
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    path = future.result()
                    results[name] = path
                except Exception as e:
                    logger.error("Failed to download %s: %s", name, e)
                    errors[name] = e
    else:
        for name in names:
            try:
                path = DOWNLOADERS[name](datasets_dir, force)
                results[name] = path
            except Exception as e:
                logger.error("Failed to download %s: %s", name, e)
                errors[name] = e

    if errors:
        error_msgs = [f"  {name}: {err}" for name, err in errors.items()]
        raise RuntimeError(
            f"Dataset download failed for {len(errors)} dataset(s):\n"
            + "\n".join(error_msgs)
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for dataset downloads."""
    parser = argparse.ArgumentParser(
        description="Download evaluation datasets for the CUMRAG benchmark harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Datasets:\n"
            "  rgb       RGB benchmark (git clone)\n"
            "  nq        Natural Questions / nq_open (HuggingFace)\n"
            "  halueval  HaluEval (git clone)\n"
            "\n"
            "Examples:\n"
            "  python scripts/download_datasets.py                  # Download all\n"
            "  python scripts/download_datasets.py --datasets rgb nq  # Download specific\n"
            "  python scripts/download_datasets.py --force           # Force re-download\n"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DOWNLOADERS.keys()),
        default=None,
        help="Which datasets to download (default: all)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help=f"Directory to download into (default: {DEFAULT_DATASETS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if datasets already exist",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Download datasets sequentially instead of in parallel",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)

    with timer("Total dataset download", logger):
        results = download_all(
            datasets_dir=args.datasets_dir,
            datasets=args.datasets,
            force=args.force,
            parallel=not args.no_parallel,
        )

    print("\nDatasets downloaded:")
    for name, path in sorted(results.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
