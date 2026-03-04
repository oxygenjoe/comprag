#!/usr/bin/env python3
"""Dataset statistics and summary report for the CUMRAG eval harness.

Generates a diagnostic report covering:
1. Raw data status — existence of downloaded dataset directories and key files
2. Normalized data — record counts per dataset/subset from normalized JSONL files
3. ChromaDB index — entry count, embedding model, chunk params, created_at per collection
4. Schema conformance — pass/fail per normalized JSONL file (delegates to validate_datasets)
5. Summary — totals and any gaps in the pipeline

Fast: reads only counts and metadata, never loads full datasets.

Standalone:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --datasets-dir datasets --index-dir index

Importable:
    from scripts.dataset_stats import generate_report, DatasetReport
    report = generate_report()
    report.print_tables()
"""

import argparse
import json
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
DEFAULT_INDEX_DIR = _PROJECT_ROOT / "index"

# Known datasets and their expected raw data indicators
KNOWN_DATASETS = {
    "rgb": {
        "subsets": [
            "noise_robustness",
            "negative_rejection",
            "information_integration",
            "counterfactual_robustness",
        ],
        "raw_indicators": [
            "data/en_refine.json",
            "data/en.json",
            "data/en_int.json",
            "data/en_fact.json",
        ],
    },
    "nq": {
        "subsets": ["test"],
        "raw_indicators": [
            "validation.jsonl",
            "train.jsonl",
        ],
    },
    "halueval": {
        "subsets": ["qa"],
        "raw_indicators": [
            "data/qa_data.json",
        ],
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RawDataStatus:
    """Status of raw (downloaded) data for a single dataset."""

    dataset: str
    dir_exists: bool = False
    files_found: list[str] = field(default_factory=list)
    files_missing: list[str] = field(default_factory=list)


@dataclass
class NormalizedSubsetInfo:
    """Record count for a single normalized subset."""

    dataset: str
    subset: str
    file_path: str
    record_count: int = 0
    file_exists: bool = False


@dataclass
class CollectionInfo:
    """Metadata for a single ChromaDB collection."""

    name: str
    count: int = 0
    embedding_model: str = ""
    chunk_size_words: int = 0
    chunk_overlap_words: int = 0
    created_at: str = ""
    dataset: str = ""
    comprag_version: str = ""


@dataclass
class SchemaStatus:
    """Schema validation status for a single normalized file."""

    dataset: str
    subset: str
    total_records: int = 0
    valid_records: int = 0
    violation_count: int = 0
    passed: bool = True


@dataclass
class DatasetReport:
    """Aggregated report across all datasets and indexes."""

    raw_data: list[RawDataStatus] = field(default_factory=list)
    normalized: list[NormalizedSubsetInfo] = field(default_factory=list)
    collections: list[CollectionInfo] = field(default_factory=list)
    schema: list[SchemaStatus] = field(default_factory=list)

    @property
    def total_normalized_records(self) -> int:
        return sum(n.record_count for n in self.normalized)

    @property
    def total_indexed_entries(self) -> int:
        return sum(c.count for c in self.collections)

    @property
    def schema_all_pass(self) -> bool:
        return all(s.passed for s in self.schema)

    def print_tables(self) -> None:
        """Print all report sections as formatted tables to stdout."""
        _print_raw_data_table(self.raw_data)
        _print_normalized_table(self.normalized)
        _print_collections_table(self.collections)
        _print_schema_table(self.schema)
        _print_summary(self)


# ---------------------------------------------------------------------------
# Data collection functions
# ---------------------------------------------------------------------------


def _count_jsonl_lines(file_path: Path) -> int:
    """Count non-empty lines in a JSONL file without loading data."""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def collect_raw_data_status(datasets_dir: Path) -> list[RawDataStatus]:
    """Check which raw dataset directories and key files exist.

    Args:
        datasets_dir: Root datasets directory.

    Returns:
        List of RawDataStatus, one per known dataset.
    """
    results: list[RawDataStatus] = []

    for name, info in KNOWN_DATASETS.items():
        status = RawDataStatus(dataset=name)
        ds_dir = datasets_dir / name

        if ds_dir.is_dir():
            status.dir_exists = True
            for indicator in info["raw_indicators"]:
                path = ds_dir / indicator
                if path.is_file():
                    status.files_found.append(indicator)
                else:
                    status.files_missing.append(indicator)
        else:
            status.files_missing = list(info["raw_indicators"])

        results.append(status)

    return results


def collect_normalized_info(datasets_dir: Path) -> list[NormalizedSubsetInfo]:
    """Count records in each normalized JSONL file.

    Args:
        datasets_dir: Root datasets directory.

    Returns:
        List of NormalizedSubsetInfo, one per expected subset.
    """
    results: list[NormalizedSubsetInfo] = []

    for name, info in KNOWN_DATASETS.items():
        for subset in info["subsets"]:
            jsonl_path = datasets_dir / name / "normalized" / f"{subset}.jsonl"
            entry = NormalizedSubsetInfo(
                dataset=name,
                subset=subset,
                file_path=str(jsonl_path),
            )

            if jsonl_path.is_file():
                entry.file_exists = True
                entry.record_count = _count_jsonl_lines(jsonl_path)

            results.append(entry)

    return results


def collect_collection_info(index_dir: Path) -> list[CollectionInfo]:
    """Read metadata and entry counts from all ChromaDB collections.

    Does NOT load embeddings or run queries -- metadata and count() only.

    Args:
        index_dir: Path to ChromaDB persistence directory.

    Returns:
        List of CollectionInfo, one per collection. Empty list if index_dir
        does not exist or contains no collections.
    """
    if not index_dir.is_dir():
        return []

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(index_dir))
    except Exception:
        return []

    # Extract collection names (handle both old and new ChromaDB API)
    raw_collections = client.list_collections()
    names: list[str] = []
    for c in raw_collections:
        if isinstance(c, str):
            names.append(c)
        elif hasattr(c, "name"):
            names.append(c.name)
    names.sort()

    results: list[CollectionInfo] = []

    for name in names:
        try:
            collection = client.get_collection(name)
            meta = collection.metadata or {}

            ci = CollectionInfo(
                name=name,
                count=collection.count(),
                embedding_model=str(meta.get("embedding_model", "")),
                chunk_size_words=int(meta.get("chunk_size_words", 0)),
                chunk_overlap_words=int(meta.get("chunk_overlap_words", 0)),
                created_at=str(meta.get("created_at", "")),
                dataset=str(meta.get("dataset", "")),
                comprag_version=str(meta.get("comprag_version", "")),
            )
            results.append(ci)
        except Exception:
            # Collection exists but can't be read -- record what we can
            results.append(CollectionInfo(name=name))

    return results


def collect_schema_status(datasets_dir: Path) -> list[SchemaStatus]:
    """Run schema validation on each normalized JSONL file.

    Imports the validate_datasets module to reuse existing validation logic.

    Args:
        datasets_dir: Root datasets directory.

    Returns:
        List of SchemaStatus, one per normalized file found.
    """
    try:
        from scripts.validate_datasets import validate_jsonl_file
    except ImportError:
        # Fallback: try importing from the script directly
        validate_path = _SCRIPT_DIR / "validate_datasets.py"
        if validate_path.is_file():
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "validate_datasets", str(validate_path)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            validate_jsonl_file = mod.validate_jsonl_file
        else:
            # Cannot validate -- return empty
            return []

    results: list[SchemaStatus] = []

    for name, info in KNOWN_DATASETS.items():
        for subset in info["subsets"]:
            jsonl_path = datasets_dir / name / "normalized" / f"{subset}.jsonl"
            if not jsonl_path.is_file():
                continue

            vr = validate_jsonl_file(jsonl_path)
            results.append(
                SchemaStatus(
                    dataset=name,
                    subset=subset,
                    total_records=vr.total_records,
                    valid_records=vr.valid_records,
                    violation_count=vr.violation_count,
                    passed=vr.is_valid,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_PASS = "PASS"
_FAIL = "FAIL"
_MISSING = "MISSING"


def _print_raw_data_table(raw_data: list[RawDataStatus]) -> None:
    """Print raw data status table."""
    print()
    print("=" * 70)
    print("1. Raw Data Status")
    print("=" * 70)

    col_ds = max(len(r.dataset) for r in raw_data) if raw_data else 10
    col_ds = max(col_ds, len("Dataset"))

    header = f"  {'Dataset':<{col_ds}}  {'Dir Exists':>10}  {'Files Found':>11}  {'Files Missing':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in raw_data:
        dir_status = "Yes" if r.dir_exists else "No"
        found = len(r.files_found)
        missing = len(r.files_missing)
        total = found + missing
        found_str = f"{found}/{total}"
        print(
            f"  {r.dataset:<{col_ds}}  {dir_status:>10}  "
            f"{found_str:>11}  {missing:>13}"
        )

    print()


def _print_normalized_table(normalized: list[NormalizedSubsetInfo]) -> None:
    """Print normalized dataset record counts."""
    print("=" * 70)
    print("2. Normalized Data (Record Counts)")
    print("=" * 70)

    if not normalized:
        print("  No normalized datasets found.")
        print()
        return

    col_ds = max(len(n.dataset) for n in normalized)
    col_ds = max(col_ds, len("Dataset"))
    col_sub = max(len(n.subset) for n in normalized)
    col_sub = max(col_sub, len("Subset"))

    header = f"  {'Dataset':<{col_ds}}  {'Subset':<{col_sub}}  {'Records':>8}  {'Status':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for n in normalized:
        if n.file_exists:
            status = f"{n.record_count:>8}"
            label = _PASS
        else:
            status = f"{'--':>8}"
            label = _MISSING
        print(f"  {n.dataset:<{col_ds}}  {n.subset:<{col_sub}}  {status}  {label:>8}")

    total = sum(n.record_count for n in normalized)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<{col_ds}}  {'':<{col_sub}}  {total:>8}")
    print()


def _print_collections_table(collections: list[CollectionInfo]) -> None:
    """Print ChromaDB collection metadata."""
    print("=" * 70)
    print("3. ChromaDB Collections")
    print("=" * 70)

    if not collections:
        print("  No ChromaDB collections found.")
        print()
        return

    col_name = max(len(c.name) for c in collections)
    col_name = max(col_name, len("Collection"))

    header = (
        f"  {'Collection':<{col_name}}  "
        f"{'Entries':>8}  "
        f"{'Embed Model':<20}  "
        f"{'Chunk':>5}  "
        f"{'Overlap':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for c in collections:
        model_display = c.embedding_model[:20] if c.embedding_model else "--"
        chunk_display = str(c.chunk_size_words) if c.chunk_size_words else "--"
        overlap_display = str(c.chunk_overlap_words) if c.chunk_overlap_words else "--"
        print(
            f"  {c.name:<{col_name}}  "
            f"{c.count:>8,}  "
            f"{model_display:<20}  "
            f"{chunk_display:>5}  "
            f"{overlap_display:>7}"
        )

    total = sum(c.count for c in collections)
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'TOTAL':<{col_name}}  "
        f"{total:>8,}  "
        f"{'':<20}  "
        f"{'':>5}  "
        f"{'':>7}"
    )

    # Detail section: created_at and dataset tags
    has_detail = any(c.created_at or c.dataset for c in collections)
    if has_detail:
        print()
        print("  Collection Details:")
        for c in collections:
            parts = []
            if c.dataset:
                parts.append(f"dataset={c.dataset}")
            if c.created_at:
                parts.append(f"created={c.created_at}")
            if c.comprag_version:
                parts.append(f"version={c.comprag_version}")
            if parts:
                print(f"    {c.name}: {', '.join(parts)}")

    print()


def _print_schema_table(schema: list[SchemaStatus]) -> None:
    """Print schema validation results."""
    print("=" * 70)
    print("4. Schema Conformance")
    print("=" * 70)

    if not schema:
        print("  No normalized files to validate.")
        print()
        return

    col_ds = max(len(s.dataset) for s in schema)
    col_ds = max(col_ds, len("Dataset"))
    col_sub = max(len(s.subset) for s in schema)
    col_sub = max(col_sub, len("Subset"))

    header = (
        f"  {'Dataset':<{col_ds}}  "
        f"{'Subset':<{col_sub}}  "
        f"{'Records':>8}  "
        f"{'Valid':>6}  "
        f"{'Violations':>10}  "
        f"{'Status':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for s in schema:
        status = _PASS if s.passed else _FAIL
        print(
            f"  {s.dataset:<{col_ds}}  "
            f"{s.subset:<{col_sub}}  "
            f"{s.total_records:>8}  "
            f"{s.valid_records:>6}  "
            f"{s.violation_count:>10}  "
            f"{status:>6}"
        )

    print()


def _print_summary(report: DatasetReport) -> None:
    """Print high-level summary."""
    print("=" * 70)
    print("5. Summary")
    print("=" * 70)

    print(f"  Total normalized records:  {report.total_normalized_records:>10,}")
    print(f"  Total indexed entries:     {report.total_indexed_entries:>10,}")
    print(f"  ChromaDB collections:      {len(report.collections):>10}")
    schema_status = _PASS if report.schema_all_pass else _FAIL
    print(f"  Schema conformance:        {schema_status:>10}")

    # Check for gaps
    gaps: list[str] = []
    for n in report.normalized:
        if not n.file_exists:
            gaps.append(f"{n.dataset}/{n.subset}: normalized file missing")
    for r in report.raw_data:
        if not r.dir_exists:
            gaps.append(f"{r.dataset}: raw data directory missing")
    for s in report.schema:
        if not s.passed:
            gaps.append(
                f"{s.dataset}/{s.subset}: {s.violation_count} schema violation(s)"
            )

    if gaps:
        print()
        print("  Gaps detected:")
        for g in gaps:
            print(f"    - {g}")
    else:
        print()
        print("  No gaps detected. All datasets normalized, indexed, and conformant.")

    print()


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------


def generate_report(
    datasets_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
) -> DatasetReport:
    """Generate the full dataset statistics report.

    Collects raw data status, normalized record counts, ChromaDB collection
    metadata, and schema validation results. Fast -- metadata and line counts
    only, no full data loading.

    Args:
        datasets_dir: Root datasets directory (default: <project>/datasets/).
        index_dir: ChromaDB persistence directory (default: <project>/index/).

    Returns:
        DatasetReport with all sections populated.
    """
    if datasets_dir is None:
        datasets_dir = DEFAULT_DATASETS_DIR
    datasets_dir = Path(datasets_dir)

    if index_dir is None:
        index_dir = DEFAULT_INDEX_DIR
    index_dir = Path(index_dir)

    report = DatasetReport()
    report.raw_data = collect_raw_data_status(datasets_dir)
    report.normalized = collect_normalized_info(datasets_dir)
    report.collections = collect_collection_info(index_dir)
    report.schema = collect_schema_status(datasets_dir)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point.

    Returns:
        0 on success (all checks pass), 1 if any gaps detected.
    """
    parser = argparse.ArgumentParser(
        description="Generate dataset statistics and summary report for the CUMRAG eval harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sections reported:\n"
            "  1. Raw Data Status        — downloaded dataset directories and key files\n"
            "  2. Normalized Data         — record counts per dataset/subset\n"
            "  3. ChromaDB Collections    — entry count, embedding model, chunk params\n"
            "  4. Schema Conformance      — pass/fail per normalized file\n"
            "  5. Summary                 — totals and gap detection\n"
            "\n"
            "Examples:\n"
            "  python scripts/dataset_stats.py\n"
            "  python scripts/dataset_stats.py --datasets-dir /path/to/datasets\n"
            "  python scripts/dataset_stats.py --index-dir /path/to/index\n"
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help=f"Root datasets directory (default: {DEFAULT_DATASETS_DIR})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help=f"ChromaDB persistence directory (default: {DEFAULT_INDEX_DIR})",
    )

    args = parser.parse_args()

    report = generate_report(
        datasets_dir=args.datasets_dir,
        index_dir=args.index_dir,
    )

    report.print_tables()

    # Exit 0 if no gaps, 1 if gaps found
    has_gaps = (
        any(not n.file_exists for n in report.normalized)
        or any(not r.dir_exists for r in report.raw_data)
        or any(not s.passed for s in report.schema)
    )
    return 1 if has_gaps else 0


if __name__ == "__main__":
    sys.exit(main())
