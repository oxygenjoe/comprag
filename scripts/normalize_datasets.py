#!/usr/bin/env python3
"""Normalize evaluation datasets into the unified CUMRAG JSONL schema.

Reads raw downloaded datasets (from download_datasets.py) and writes
normalized JSONL files per subset under datasets/{name}/normalized/.

Unified schema per line:
    sample_id, dataset, subset, query, ground_truth, corpus_doc_ids, metadata

Runnable standalone (CLI with argparse) AND importable as module.

Usage:
    python scripts/normalize_datasets.py --all
    python scripts/normalize_datasets.py --datasets rgb nq
    python scripts/normalize_datasets.py --datasets halueval --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure project root is importable when run standalone
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from comprag.utils import get_logger, setup_logging, timer

logger = get_logger("comprag.normalize_datasets")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASETS_DIR = _PROJECT_ROOT / "datasets"

# RGB subset -> source file mapping (English only).
# Prefer en_refine.json over en.json when available (refined = corrected labels).
#
# IMPORTANT: noise_robustness and negative_rejection share the SAME source data.
# The RGB benchmark does NOT have separate data files per subset for these two.
# All 300 entries have both "positive" and "negative" passage lists.
# The distinction is an EVAL-TIME parameter (noise_rate):
#   - noise_robustness: noise_rate < 1 (mix of positive + negative passages)
#   - negative_rejection: noise_rate = 1 (ALL passages are irrelevant; model should refuse)
# See: evalue.py lines 50-65, and readme.md line 90.
# We normalize both from the same file but tag each with its eval-time noise_rate
# so the evaluation pipeline knows how to construct passage sets.
RGB_SUBSET_FILES: dict[str, list[str]] = {
    "noise_robustness": ["en_refine.json", "en.json"],
    "negative_rejection": ["en_refine.json", "en.json"],
    "information_integration": ["en_int.json"],
    "counterfactual_robustness": ["en_fact.json"],
}

# Eval-time noise_rate per subset. This determines how passages are assembled:
# noise_rate < 1 means some passages are relevant; noise_rate = 1 means ALL are noise.
RGB_SUBSET_NOISE_RATE: dict[str, float] = {
    "noise_robustness": 0.5,       # default eval noise_rate (mix of positive + negative)
    "negative_rejection": 1.0,     # all passages are irrelevant — model must refuse
    "information_integration": 0.0, # all passages are relevant (multi-doc integration)
    "counterfactual_robustness": 0.0, # passages contain factual errors to detect
}

# HaluEval QA data file (relative to halueval dataset dir)
HALUEVAL_QA_PATHS = ["data/qa_data.json"]

# NQ split to use as the "test" subset
NQ_TEST_SPLIT = "validation"


# ---------------------------------------------------------------------------
# Per-dataset transform functions
# ---------------------------------------------------------------------------


def _extract_rgb_answer(raw_answer: object) -> tuple[str, list[str]]:
    """Extract primary answer string and all alternatives from RGB answer field.

    RGB raw data has ``answer`` as a nested structure:
    - List of lists: ``[["alt1", "alt2", ...], ["alt3", ...]]``
    - List of mixed strings and lists: ``[["alt1", "alt2"], "alt3"]``
    - List of strings: ``["alt1", "alt2"]``
    - Plain string: ``"answer"``

    Returns:
        Tuple of (primary_answer_string, flat_list_of_all_alternatives).
    """
    if isinstance(raw_answer, str):
        return raw_answer, [raw_answer]

    if isinstance(raw_answer, list):
        # Flatten: each element is either a string or a list of strings
        all_answers: list[str] = []
        for item in raw_answer:
            if isinstance(item, str):
                all_answers.append(item)
            elif isinstance(item, list):
                all_answers.extend(str(s) for s in item)
            else:
                all_answers.append(str(item))

        if all_answers:
            return all_answers[0], all_answers
        return "", []

    # Unexpected type — coerce to string
    return str(raw_answer), [str(raw_answer)]


def normalize_rgb(raw_entry: dict, subset: str, index: int) -> dict:
    """Transform a single RGB entry into the unified schema.

    Args:
        raw_entry: Raw JSON object from the RGB data file.
            Expected fields: question, answer, passages (list), label (optional).
        subset: RGB subset name (e.g. 'noise_robustness').
        index: Zero-based index within the subset.

    Returns:
        Dict matching the unified JSONL schema.
    """
    # RGB data uses "query" not "question", and "positive"/"negative" not "passages"
    question = raw_entry.get("question") or raw_entry["query"]
    passages = raw_entry.get("passages") or (
        raw_entry.get("positive", []) + raw_entry.get("negative", [])
    )
    primary_answer, all_answers = _extract_rgb_answer(raw_entry["answer"])
    return {
        "sample_id": f"rgb_{subset}_{index:04d}",
        "dataset": "rgb",
        "subset": subset,
        "query": question,
        "ground_truth": primary_answer,
        "corpus_doc_ids": [],  # passages are injected into ChromaDB at index time
        "metadata": {
            "all_answers": all_answers,
            "original_passages": passages,
            "label": raw_entry.get("label", None),
            # Eval-time noise_rate distinguishes how passages are assembled:
            # noise_robustness (< 1): mix of positive + negative passages
            # negative_rejection (= 1): ALL passages are irrelevant noise
            "noise_rate": RGB_SUBSET_NOISE_RATE.get(subset, 0.0),
        },
    }


def normalize_nq(raw_entry: dict, index: int) -> dict:
    """Transform a single Natural Questions entry into the unified schema.

    Args:
        raw_entry: Raw JSON object from the nq_open JSONL file.
            Expected fields: question, answer (list of strings).
        index: Zero-based index within the subset.

    Returns:
        Dict matching the unified JSONL schema.
    """
    answers = raw_entry["answer"]
    return {
        "sample_id": f"nq_test_{index:04d}",
        "dataset": "nq",
        "subset": "test",
        "query": raw_entry["question"],
        "ground_truth": answers[0],  # primary answer
        "corpus_doc_ids": [],  # retrieves from global Wikipedia index
        "metadata": {
            "all_answers": answers,
        },
    }


def normalize_halueval(raw_entry: dict, index: int) -> dict:
    """Transform a single HaluEval QA entry into the unified schema.

    Args:
        raw_entry: Raw JSON object from HaluEval qa_data.json.
            Expected fields: knowledge, question, right_answer (or answer),
            hallucinated_answer.
        index: Zero-based index within the subset.

    Returns:
        Dict matching the unified JSONL schema.
    """
    # HaluEval uses 'right_answer' in the actual data; the spec references
    # 'answer'. Handle both for robustness.
    ground_truth = raw_entry.get("right_answer") or raw_entry.get("answer", "")
    return {
        "sample_id": f"halueval_qa_{index:04d}",
        "dataset": "halueval",
        "subset": "qa",
        "query": raw_entry["question"],
        "ground_truth": ground_truth,
        "corpus_doc_ids": [],
        "metadata": {
            "knowledge": raw_entry["knowledge"],
            "hallucinated_answer": raw_entry["hallucinated_answer"],
        },
    }


# ---------------------------------------------------------------------------
# Dataset-level normalization orchestrators
# ---------------------------------------------------------------------------


def _find_rgb_data_dir(datasets_dir: Path) -> Path:
    """Locate the RGB data/ directory within the downloaded repo.

    The git clone puts repo contents directly in datasets/rgb/, so the
    data files are at datasets/rgb/data/*.json.

    Returns:
        Path to the data/ directory.

    Raises:
        FileNotFoundError: If the data directory is not found.
    """
    candidate = datasets_dir / "rgb" / "data"
    if candidate.is_dir():
        return candidate

    # Fallback: maybe the clone created a nested RGB/ directory
    nested = datasets_dir / "rgb" / "RGB" / "data"
    if nested.is_dir():
        return nested

    raise FileNotFoundError(
        f"RGB data directory not found. Looked in:\n"
        f"  {candidate}\n"
        f"  {nested}\n"
        f"Run 'python scripts/download_datasets.py --datasets rgb' first."
    )


def _resolve_rgb_file(data_dir: Path, candidates: list[str]) -> Path:
    """Find the first existing file from a list of candidates.

    Args:
        data_dir: Directory containing the data files.
        candidates: Filenames to try in order of preference.

    Returns:
        Path to the first existing file.

    Raises:
        FileNotFoundError: If none of the candidates exist.
    """
    for name in candidates:
        path = data_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"None of the expected RGB data files found in {data_dir}: {candidates}"
    )


def _load_json_or_jsonl(path: Path) -> list[dict]:
    """Load data from a JSON file (array or JSONL).

    Handles:
    - Standard JSON array: [{"a": 1}, {"b": 2}]
    - JSONL: one JSON object per line
    - JSON array of lines where each line has been serialized as a string

    Args:
        path: Path to the file.

    Returns:
        List of dicts.
    """
    text = path.read_text(encoding="utf-8").strip()

    # Try parsing as a single JSON document first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        # Single object — wrap in list
        return [data]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL (one JSON object per line)
    records = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse line {line_num} in {path}: {e}"
            ) from e
    return records


def process_rgb(
    datasets_dir: Path,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Normalize all RGB subsets.

    Args:
        datasets_dir: Root datasets directory.
        output_dir: Override output directory. Default: datasets/rgb/normalized/

    Returns:
        Dict mapping subset name -> output JSONL path.
    """
    data_dir = _find_rgb_data_dir(datasets_dir)
    if output_dir is None:
        output_dir = datasets_dir / "rgb" / "normalized"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    for subset, file_candidates in RGB_SUBSET_FILES.items():
        source_file = _resolve_rgb_file(data_dir, file_candidates)
        logger.info("RGB %s: loading from %s", subset, source_file)

        raw_data = _load_json_or_jsonl(source_file)
        logger.info("RGB %s: %d raw entries", subset, len(raw_data))

        output_path = output_dir / f"{subset}.jsonl"
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, entry in enumerate(raw_data):
                try:
                    normalized = normalize_rgb(entry, subset, idx)
                    line = json.dumps(
                        normalized, ensure_ascii=False, separators=(",", ":")
                    )
                    f.write(line + "\n")
                    count += 1
                except (KeyError, TypeError) as e:
                    logger.warning(
                        "RGB %s: skipping entry %d: %s", subset, idx, e
                    )

        logger.info("RGB %s: wrote %d entries to %s", subset, count, output_path)
        results[subset] = output_path

    return results


def process_nq(
    datasets_dir: Path,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Normalize the Natural Questions dataset.

    Uses the validation split as the 'test' subset (nq_open has no
    official test split — validation is used for evaluation).

    Args:
        datasets_dir: Root datasets directory.
        output_dir: Override output directory. Default: datasets/nq/normalized/

    Returns:
        Dict mapping subset name -> output JSONL path.
    """
    nq_dir = datasets_dir / "nq"
    if output_dir is None:
        output_dir = datasets_dir / "nq" / "normalized"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the source split file
    source_file = nq_dir / f"{NQ_TEST_SPLIT}.jsonl"
    if not source_file.is_file():
        # Try alternate naming
        alt = nq_dir / f"{NQ_TEST_SPLIT}.json"
        if alt.is_file():
            source_file = alt
        else:
            raise FileNotFoundError(
                f"NQ source file not found: {source_file}\n"
                f"Run 'python scripts/download_datasets.py --datasets nq' first."
            )

    logger.info("NQ test: loading from %s", source_file)

    output_path = output_dir / "test.jsonl"
    count = 0
    with open(source_file, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                normalized = normalize_nq(entry, idx)
                out_line = json.dumps(
                    normalized, ensure_ascii=False, separators=(",", ":")
                )
                fout.write(out_line + "\n")
                count += 1
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("NQ test: skipping entry %d: %s", idx, e)

    logger.info("NQ test: wrote %d entries to %s", count, output_path)
    return {"test": output_path}


def _find_halueval_qa(datasets_dir: Path) -> Path:
    """Locate the HaluEval QA data file.

    HaluEval is cloned from GitHub. The QA data file is at
    datasets/halueval/data/qa_data.json.

    Returns:
        Path to the QA data file.

    Raises:
        FileNotFoundError: If not found.
    """
    candidates = [
        datasets_dir / "halueval" / "data" / "qa_data.json",
        datasets_dir / "halueval" / "HaluEval" / "data" / "qa_data.json",
        # Some versions may put it at the repo root
        datasets_dir / "halueval" / "qa_data.json",
    ]
    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"HaluEval QA data file not found. Looked in:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nRun 'python scripts/download_datasets.py --datasets halueval' first."
    )


def process_halueval(
    datasets_dir: Path,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Normalize the HaluEval QA dataset.

    Args:
        datasets_dir: Root datasets directory.
        output_dir: Override output directory. Default: datasets/halueval/normalized/

    Returns:
        Dict mapping subset name -> output JSONL path.
    """
    qa_file = _find_halueval_qa(datasets_dir)
    if output_dir is None:
        output_dir = datasets_dir / "halueval" / "normalized"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("HaluEval QA: loading from %s", qa_file)

    raw_data = _load_json_or_jsonl(qa_file)
    logger.info("HaluEval QA: %d raw entries", len(raw_data))

    output_path = output_dir / "qa.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(raw_data):
            try:
                normalized = normalize_halueval(entry, idx)
                line = json.dumps(
                    normalized, ensure_ascii=False, separators=(",", ":")
                )
                f.write(line + "\n")
                count += 1
            except (KeyError, TypeError) as e:
                logger.warning(
                    "HaluEval QA: skipping entry %d: %s", idx, e
                )

    logger.info("HaluEval QA: wrote %d entries to %s", count, output_path)
    return {"qa": output_path}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

NORMALIZERS = {
    "rgb": process_rgb,
    "nq": process_nq,
    "halueval": process_halueval,
}


def normalize_all(
    datasets_dir: Optional[Path] = None,
    datasets: Optional[list[str]] = None,
) -> dict[str, dict[str, Path]]:
    """Normalize all (or selected) datasets into unified JSONL.

    Args:
        datasets_dir: Root datasets directory (default: <project>/datasets/).
        datasets: List of dataset names to normalize. None means all.

    Returns:
        Nested dict: {dataset_name: {subset_name: output_path}}.

    Raises:
        ValueError: If unknown dataset names are requested.
        FileNotFoundError: If raw data is missing (not downloaded).
    """
    if datasets_dir is None:
        datasets_dir = DEFAULT_DATASETS_DIR
    datasets_dir = Path(datasets_dir)

    names = datasets if datasets else list(NORMALIZERS.keys())
    invalid = set(names) - set(NORMALIZERS.keys())
    if invalid:
        raise ValueError(
            f"Unknown dataset(s): {invalid}. Valid: {list(NORMALIZERS.keys())}"
        )

    results: dict[str, dict[str, Path]] = {}
    errors: dict[str, Exception] = {}

    for name in names:
        try:
            with timer(f"Normalize {name}", logger):
                results[name] = NORMALIZERS[name](datasets_dir)
        except Exception as e:
            logger.error("Failed to normalize %s: %s", name, e)
            errors[name] = e

    if errors:
        error_msgs = [f"  {name}: {err}" for name, err in errors.items()]
        raise RuntimeError(
            f"Normalization failed for {len(errors)} dataset(s):\n"
            + "\n".join(error_msgs)
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for dataset normalization."""
    parser = argparse.ArgumentParser(
        description="Normalize evaluation datasets into unified CUMRAG JSONL schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Datasets:\n"
            "  rgb       RGB benchmark (4 subsets)\n"
            "  nq        Natural Questions / nq_open (test subset)\n"
            "  halueval  HaluEval QA (qa subset)\n"
            "\n"
            "Output schema (per JSONL line):\n"
            "  sample_id, dataset, subset, query, ground_truth,\n"
            "  corpus_doc_ids, metadata\n"
            "\n"
            "Examples:\n"
            "  python scripts/normalize_datasets.py --all\n"
            "  python scripts/normalize_datasets.py --datasets rgb nq\n"
            "  python scripts/normalize_datasets.py --datasets halueval -v\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Normalize all datasets",
    )
    group.add_argument(
        "--datasets",
        nargs="+",
        choices=list(NORMALIZERS.keys()),
        help="Which datasets to normalize",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help=f"Root datasets directory (default: {DEFAULT_DATASETS_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)

    selected = None if args.all else args.datasets

    with timer("Total normalization", logger):
        results = normalize_all(
            datasets_dir=args.datasets_dir,
            datasets=selected,
        )

    print("\nNormalized datasets:")
    for dataset_name, subsets in sorted(results.items()):
        for subset_name, path in sorted(subsets.items()):
            print(f"  {dataset_name}/{subset_name}: {path}")


if __name__ == "__main__":
    main()
