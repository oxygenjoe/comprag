#!/usr/bin/env python3
"""Download benchmark datasets for CompRAG evaluation.

Supports RGB, NQ (Natural Questions), and HaluEval.
Idempotent: skips datasets that are already downloaded.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

RGB_REPO = "https://github.com/chen700564/RGB.git"
HALUEVAL_REPO = "https://github.com/RUCAIBox/HaluEval.git"
NQ_HF_DATASET = "nq_open"
NQ_HF_BASE_URL = (
    "https://huggingface.co/datasets/google-research-datasets/natural_questions"
)


def _run_git_clone(repo_url: str, dest: Path) -> None:
    """Clone a git repository to the destination path."""
    logger.info("Cloning %s -> %s", repo_url, dest)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Clone complete: %s", dest)
    except subprocess.CalledProcessError as e:
        logger.error("git clone failed for %s: %s", repo_url, e.stderr)
        raise


def _is_downloaded(dest: Path) -> bool:
    """Check if a dataset directory exists and has content."""
    if not dest.exists():
        return False
    # Check for .git dir (cloned repos) or marker file (HF downloads)
    has_git = (dest / ".git").is_dir()
    has_marker = (dest / "_download_complete").is_file()
    has_content = any(dest.iterdir())
    return has_git or has_marker or has_content


def download_rgb(base_dir: Path) -> None:
    """Download the RGB (Reasoning over Grounded Beliefs) dataset from GitHub."""
    dest = base_dir / "rgb"
    if _is_downloaded(dest):
        logger.info("RGB already downloaded at %s — skipping", dest)
        return

    dest.mkdir(parents=True, exist_ok=True)
    _run_git_clone(RGB_REPO, dest)
    logger.info("RGB download complete")


def download_halueval(base_dir: Path) -> None:
    """Download the HaluEval dataset from GitHub."""
    dest = base_dir / "halueval"
    if _is_downloaded(dest):
        logger.info("HaluEval already downloaded at %s — skipping", dest)
        return

    dest.mkdir(parents=True, exist_ok=True)
    _run_git_clone(HALUEVAL_REPO, dest)
    logger.info("HaluEval download complete")


def _download_nq_split(
    split: str, dest: Path, base_url: str
) -> None:
    """Download a single NQ split from HuggingFace."""
    url = (
        f"https://huggingface.co/datasets/google-research-datasets/"
        f"natural_questions/resolve/main/{split}.jsonl"
    )
    out_file = dest / f"{split}.jsonl"
    if out_file.exists() and out_file.stat().st_size > 0:
        logger.info("NQ %s split already exists — skipping", split)
        return

    logger.info("Downloading NQ %s split from %s", split, url)
    try:
        urllib.request.urlretrieve(url, str(out_file))
        logger.info("Downloaded %s (%.2f MB)", out_file, out_file.stat().st_size / 1e6)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        logger.error("Failed to download NQ %s: %s", split, e)
        # Clean up partial download
        if out_file.exists():
            out_file.unlink()
        raise


def _download_nq_hf_datasets(dest: Path) -> None:
    """Download NQ using the datasets library if available."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "The 'datasets' library is not installed. "
            "Install with: pip install datasets"
        )
        raise

    logger.info("Downloading NQ via HuggingFace datasets library")
    ds = load_dataset(NQ_HF_DATASET)

    for split_name in ds:
        out_file = dest / f"{split_name}.jsonl"
        if out_file.exists() and out_file.stat().st_size > 0:
            logger.info("NQ %s split already exists — skipping", split_name)
            continue

        logger.info("Writing NQ %s split (%d examples)", split_name, len(ds[split_name]))
        with open(out_file, "w", encoding="utf-8") as f:
            for row in ds[split_name]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("NQ download complete")


def download_nq(base_dir: Path) -> None:
    """Download the Natural Questions (nq_open) dataset from HuggingFace.

    Uses the `datasets` library (preferred) or falls back to direct URL download.
    """
    dest = base_dir / "nq"
    if _is_downloaded(dest):
        logger.info("NQ already downloaded at %s — skipping", dest)
        return

    dest.mkdir(parents=True, exist_ok=True)

    try:
        _download_nq_hf_datasets(dest)
    except Exception as e:
        logger.warning(
            "HuggingFace datasets download failed (%s), "
            "trying direct URL download",
            e,
        )
        try:
            for split in ("train", "validation"):
                _download_nq_split(split, dest, NQ_HF_BASE_URL)
        except Exception as e2:
            logger.error("All NQ download methods failed: %s", e2)
            raise

    # Write marker file
    marker = dest / "_download_complete"
    marker.write_text("done\n")
    logger.info("NQ download complete")


DATASET_DOWNLOADERS = {
    "rgb": download_rgb,
    "nq": download_nq,
    "halueval": download_halueval,
}


def download_all(base_dir: Path) -> None:
    """Download all supported datasets."""
    for name, downloader in DATASET_DOWNLOADERS.items():
        logger.info("--- Processing dataset: %s ---", name)
        try:
            downloader(base_dir)
        except Exception as e:
            logger.error("Failed to download %s: %s", name, e)
            raise


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for CompRAG evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --dataset all\n"
            "  %(prog)s --dataset rgb\n"
            "  %(prog)s --dataset nq --output-dir /tmp/datasets\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "rgb", "nq", "halueval"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASETS_DIR,
        help=f"Base output directory (default: {DATASETS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def _clean_dataset(base_dir: Path, name: str) -> None:
    """Remove a dataset directory for forced re-download."""
    dest = base_dir / name
    if dest.exists():
        logger.info("Removing existing %s for re-download", dest)
        import shutil
        shutil.rmtree(dest)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the dataset downloader."""
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_dir: Path = args.output_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", base_dir)

    if args.dataset == "all":
        if args.force:
            for name in DATASET_DOWNLOADERS:
                _clean_dataset(base_dir, name)
        try:
            download_all(base_dir)
        except Exception:
            logger.error("One or more dataset downloads failed")
            return 1
    else:
        if args.force:
            _clean_dataset(base_dir, args.dataset)
        try:
            DATASET_DOWNLOADERS[args.dataset](base_dir)
        except Exception:
            logger.error("Dataset download failed: %s", args.dataset)
            return 1

    logger.info("All requested downloads complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
