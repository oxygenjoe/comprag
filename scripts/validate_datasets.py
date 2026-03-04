#!/usr/bin/env python3
"""Validate normalized JSONL datasets against the unified CompRAG schema.

Checks every normalized JSONL record for schema conformance:
- sample_id: string, matches pattern {dataset}_{subset}_{index:04d}
- dataset: string, one of rgb/nq/halueval
- subset: string, non-empty
- query: string, non-empty
- ground_truth: string (NOT list/dict)
- corpus_doc_ids: list of strings (can be empty)
- metadata: dict

Reports all violations with file, line number, field, expected type, actual
type/value.

Runnable standalone (CLI with argparse) AND importable as module.

Usage:
    python scripts/validate_datasets.py
    python scripts/validate_datasets.py --datasets-dir /path/to/datasets
    python scripts/validate_datasets.py --verbose
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure project root is importable when run standalone
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASETS_DIR = _PROJECT_ROOT / "datasets"

VALID_DATASETS = {"rgb", "nq", "halueval"}

# Pattern: {dataset}_{subset}_{index:04d} where index is zero-padded 4+ digits
SAMPLE_ID_PATTERN = re.compile(
    r"^(?P<dataset>[a-z]+)_(?P<subset>[a-z_]+)_(?P<index>\d{4,})$"
)


# ---------------------------------------------------------------------------
# Violation tracking
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """A single schema violation."""

    file: str
    line: int
    field: str
    expected: str
    actual: str

    def __str__(self) -> str:
        return (
            f"{self.file}:{self.line} — field '{self.field}': "
            f"expected {self.expected}, got {self.actual}"
        )


@dataclass
class ValidationResult:
    """Aggregated validation results."""

    total_records: int = 0
    valid_records: int = 0
    violations: list[Violation] = field(default_factory=list)

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def is_valid(self) -> bool:
        return self.violation_count == 0


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------


def validate_record(
    record: dict,
    file_path: str,
    line_num: int,
) -> list[Violation]:
    """Validate a single JSONL record against the unified schema.

    Args:
        record: Parsed JSON object from a JSONL line.
        file_path: Path to the source file (for error reporting).
        line_num: 1-based line number (for error reporting).

    Returns:
        List of Violation objects (empty if the record is valid).
    """
    violations: list[Violation] = []

    def _add(field_name: str, expected: str, actual: str) -> None:
        violations.append(
            Violation(
                file=file_path,
                line=line_num,
                field=field_name,
                expected=expected,
                actual=actual,
            )
        )

    # --- sample_id ---
    sample_id = record.get("sample_id")
    if sample_id is None:
        _add("sample_id", "string matching {dataset}_{subset}_{index:04d}", "missing")
    elif not isinstance(sample_id, str):
        _add(
            "sample_id",
            "string matching {dataset}_{subset}_{index:04d}",
            f"{type(sample_id).__name__}: {_truncate(sample_id)}",
        )
    else:
        m = SAMPLE_ID_PATTERN.match(sample_id)
        if not m:
            _add(
                "sample_id",
                "string matching {dataset}_{subset}_{index:04d}",
                f"'{sample_id}'",
            )

    # --- dataset ---
    dataset = record.get("dataset")
    if dataset is None:
        _add("dataset", f"string, one of {sorted(VALID_DATASETS)}", "missing")
    elif not isinstance(dataset, str):
        _add(
            "dataset",
            f"string, one of {sorted(VALID_DATASETS)}",
            f"{type(dataset).__name__}: {_truncate(dataset)}",
        )
    elif dataset not in VALID_DATASETS:
        _add(
            "dataset",
            f"string, one of {sorted(VALID_DATASETS)}",
            f"'{dataset}'",
        )

    # --- subset ---
    subset = record.get("subset")
    if subset is None:
        _add("subset", "non-empty string", "missing")
    elif not isinstance(subset, str):
        _add(
            "subset",
            "non-empty string",
            f"{type(subset).__name__}: {_truncate(subset)}",
        )
    elif not subset:
        _add("subset", "non-empty string", "empty string")

    # --- query ---
    query = record.get("query")
    if query is None:
        _add("query", "non-empty string", "missing")
    elif not isinstance(query, str):
        _add(
            "query",
            "non-empty string",
            f"{type(query).__name__}: {_truncate(query)}",
        )
    elif not query:
        _add("query", "non-empty string", "empty string")

    # --- ground_truth (must be string, NOT list/dict) ---
    ground_truth = record.get("ground_truth")
    if ground_truth is None:
        _add("ground_truth", "string", "missing")
    elif not isinstance(ground_truth, str):
        _add(
            "ground_truth",
            "string",
            f"{type(ground_truth).__name__}: {_truncate(ground_truth)}",
        )

    # --- corpus_doc_ids ---
    corpus_doc_ids = record.get("corpus_doc_ids")
    if corpus_doc_ids is None:
        _add("corpus_doc_ids", "list of strings", "missing")
    elif not isinstance(corpus_doc_ids, list):
        _add(
            "corpus_doc_ids",
            "list of strings",
            f"{type(corpus_doc_ids).__name__}: {_truncate(corpus_doc_ids)}",
        )
    else:
        for i, item in enumerate(corpus_doc_ids):
            if not isinstance(item, str):
                _add(
                    f"corpus_doc_ids[{i}]",
                    "string",
                    f"{type(item).__name__}: {_truncate(item)}",
                )

    # --- metadata ---
    metadata = record.get("metadata")
    if metadata is None:
        _add("metadata", "dict", "missing")
    elif not isinstance(metadata, dict):
        _add(
            "metadata",
            "dict",
            f"{type(metadata).__name__}: {_truncate(metadata)}",
        )

    # --- Cross-field: sample_id consistency with dataset/subset ---
    if (
        isinstance(sample_id, str)
        and isinstance(dataset, str)
        and isinstance(subset, str)
    ):
        m = SAMPLE_ID_PATTERN.match(sample_id)
        if m:
            id_dataset = m.group("dataset")
            id_subset = m.group("subset")
            if id_dataset != dataset:
                _add(
                    "sample_id",
                    f"dataset prefix '{dataset}'",
                    f"prefix '{id_dataset}' in '{sample_id}'",
                )
            if id_subset != subset:
                _add(
                    "sample_id",
                    f"subset component '{subset}'",
                    f"component '{id_subset}' in '{sample_id}'",
                )

    return violations


def _truncate(value: object, max_len: int = 80) -> str:
    """Truncate a repr for display in violation messages."""
    s = repr(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


# ---------------------------------------------------------------------------
# File / directory validation
# ---------------------------------------------------------------------------


def validate_jsonl_file(file_path: Path) -> ValidationResult:
    """Validate all records in a single JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        ValidationResult with all violations found.
    """
    result = ValidationResult()
    display_path = str(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                result.violations.append(
                    Violation(
                        file=display_path,
                        line=line_num,
                        field="<json>",
                        expected="valid JSON",
                        actual=f"JSONDecodeError: {e}",
                    )
                )
                result.total_records += 1
                continue

            result.total_records += 1
            violations = validate_record(record, display_path, line_num)
            if violations:
                result.violations.extend(violations)
            else:
                result.valid_records += 1

    return result


def find_normalized_jsonl(datasets_dir: Path) -> list[Path]:
    """Find all normalized JSONL files in the datasets directory.

    Searches for datasets/{name}/normalized/*.jsonl for each known dataset.

    Args:
        datasets_dir: Root datasets directory.

    Returns:
        Sorted list of JSONL file paths.
    """
    files: list[Path] = []
    for dataset_name in sorted(VALID_DATASETS):
        normalized_dir = datasets_dir / dataset_name / "normalized"
        if normalized_dir.is_dir():
            files.extend(sorted(normalized_dir.glob("*.jsonl")))
    return files


def validate_datasets(
    datasets_dir: Optional[Path] = None,
    verbose: bool = False,
) -> ValidationResult:
    """Validate all normalized JSONL datasets.

    Args:
        datasets_dir: Root datasets directory (default: <project>/datasets/).
        verbose: If True, print each violation as it's found.

    Returns:
        Aggregated ValidationResult.
    """
    if datasets_dir is None:
        datasets_dir = DEFAULT_DATASETS_DIR
    datasets_dir = Path(datasets_dir)

    files = find_normalized_jsonl(datasets_dir)
    if not files:
        print(f"No normalized JSONL files found in {datasets_dir}", file=sys.stderr)
        return ValidationResult()

    combined = ValidationResult()

    for file_path in files:
        result = validate_jsonl_file(file_path)
        combined.total_records += result.total_records
        combined.valid_records += result.valid_records
        combined.violations.extend(result.violations)

        status = "PASS" if result.is_valid else "FAIL"
        print(
            f"  [{status}] {file_path.relative_to(datasets_dir)}: "
            f"{result.total_records} records, "
            f"{result.violation_count} violations"
        )

        if verbose and result.violations:
            for v in result.violations:
                print(f"    {v}")

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point. Returns exit code (0=clean, 1=violations found)."""
    parser = argparse.ArgumentParser(
        description="Validate normalized JSONL datasets against the unified CompRAG schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Schema contract (per JSONL record):\n"
            "  sample_id      string, pattern {dataset}_{subset}_{index:04d}\n"
            "  dataset        string, one of rgb/nq/halueval\n"
            "  subset         string, non-empty\n"
            "  query          string, non-empty\n"
            "  ground_truth   string (NOT list/dict)\n"
            "  corpus_doc_ids list of strings (can be empty)\n"
            "  metadata       dict\n"
            "\n"
            "Examples:\n"
            "  python scripts/validate_datasets.py\n"
            "  python scripts/validate_datasets.py --datasets-dir /path/to/datasets\n"
            "  python scripts/validate_datasets.py --verbose\n"
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help=f"Root datasets directory (default: {DEFAULT_DATASETS_DIR})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each violation detail",
    )

    args = parser.parse_args()

    print("Validating normalized datasets against unified schema...\n")

    result = validate_datasets(
        datasets_dir=args.datasets_dir,
        verbose=args.verbose,
    )

    print(
        f"\nSummary: {result.total_records} records checked, "
        f"{result.valid_records} valid, "
        f"{result.violation_count} violations"
    )

    if result.is_valid:
        print("All records conform to schema.")
        return 0
    else:
        print(f"\nViolations found ({result.violation_count} total):")
        # Group violations by file
        by_file: dict[str, list[Violation]] = {}
        for v in result.violations:
            by_file.setdefault(v.file, []).append(v)

        for file_path, violations in sorted(by_file.items()):
            print(f"\n  {file_path} ({len(violations)} violations):")
            for v in violations[:10]:
                print(f"    line {v.line}: {v.field} — expected {v.expected}, got {v.actual}")
            if len(violations) > 10:
                print(f"    ... and {len(violations) - 10} more")

        return 1


if __name__ == "__main__":
    sys.exit(main())
