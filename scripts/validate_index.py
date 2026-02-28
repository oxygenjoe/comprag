#!/usr/bin/env python3
"""Audit ChromaDB index integrity for the CUMRAG eval harness.

Validates all ChromaDB collections in the index directory against
eval_config.yaml parameters, checks sample entries for valid fields,
runs test queries, and calls the existing validate_index() from
cumrag.retriever.

Standalone:
    python scripts/validate_index.py
    python scripts/validate_index.py --index-dir index --samples 10 --verbose

Importable:
    from scripts.validate_index import audit_index
    report = audit_index(index_dir="index", n_samples=5)
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so cumrag package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chromadb

from cumrag.retriever import validate_index as retriever_validate_index
from cumrag.utils import get_logger, load_config, setup_logging

logger = get_logger("cumrag.validate_index")

# ---------------------------------------------------------------------------
# Expected config keys to check in collection metadata
# ---------------------------------------------------------------------------

_METADATA_CHECKS = {
    "embedding_model": lambda cfg: cfg["retrieval"]["embedding_model"],
    "embedding_dim": lambda cfg: cfg["retrieval"]["embedding_dim"],
    "chunk_size_words": lambda cfg: cfg["retrieval"]["chunk_size"],
    "chunk_overlap_words": lambda cfg: cfg["retrieval"]["overlap"],
}

# Entry-level metadata fields: at least one of doc_id or sample_id must exist
_REQUIRED_ID_FIELDS = ("doc_id", "sample_id")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class CollectionReport:
    """Validation report for a single ChromaDB collection."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.count: int = 0
        self.metadata_ok: bool = True
        self.metadata_errors: list[str] = []
        self.sample_ok: bool = True
        self.sample_errors: list[str] = []
        self.query_ok: bool = True
        self.query_error: str = ""
        self.retriever_validate_ok: bool = True
        self.retriever_validate_error: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.metadata_ok
            and self.sample_ok
            and self.query_ok
            and self.retriever_validate_ok
        )


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def _list_collection_names(client: chromadb.ClientAPI) -> list[str]:
    """Extract collection name strings from ChromaDB client."""
    collections = client.list_collections()
    names = []
    for c in collections:
        if isinstance(c, str):
            names.append(c)
        elif hasattr(c, "name"):
            names.append(c.name)
    return sorted(names)


def _check_metadata(
    collection: chromadb.Collection, config: dict
) -> tuple[bool, list[str]]:
    """Validate collection metadata against eval_config parameters.

    Returns:
        (ok, list_of_error_strings)
    """
    meta = collection.metadata or {}
    errors: list[str] = []

    for meta_key, config_getter in _METADATA_CHECKS.items():
        expected = config_getter(config)
        actual = meta.get(meta_key)
        if actual is None:
            errors.append(f"missing metadata key '{meta_key}' (expected {expected})")
        elif actual != expected:
            errors.append(
                f"metadata '{meta_key}' mismatch: got {actual!r}, expected {expected!r}"
            )

    return (len(errors) == 0, errors)


def _check_samples(
    collection: chromadb.Collection, n_samples: int
) -> tuple[bool, list[str]]:
    """Sample N random entries and verify they have valid fields.

    Checks:
        - Each entry has at least one of doc_id or sample_id in metadata
        - Each entry has non-empty document text

    Returns:
        (ok, list_of_error_strings)
    """
    total = collection.count()
    if total == 0:
        return (False, ["collection is empty"])

    # Get all IDs, then sample randomly
    # ChromaDB get() with limit returns first N, so we use offset for randomness
    # For efficiency, get a random subset of IDs
    n = min(n_samples, total)

    # Use random offsets to sample entries spread across the collection
    random.seed(42)  # Reproducible sampling
    offsets = sorted(random.sample(range(total), n))

    errors: list[str] = []

    for offset in offsets:
        result = collection.get(
            limit=1, offset=offset, include=["documents", "metadatas"]
        )

        if not result["ids"]:
            errors.append(f"no entry returned at offset {offset}")
            continue

        entry_id = result["ids"][0]
        meta = result["metadatas"][0] if result["metadatas"] else {}
        doc = result["documents"][0] if result["documents"] else ""

        # Check ID fields
        has_id_field = any(meta.get(f) for f in _REQUIRED_ID_FIELDS)
        if not has_id_field:
            errors.append(
                f"entry '{entry_id}': missing both doc_id and sample_id in metadata"
            )

        # Check non-empty document
        if not doc or not doc.strip():
            errors.append(f"entry '{entry_id}': empty document text")

    return (len(errors) == 0, errors)


def _check_query(
    client: chromadb.ClientAPI, collection_name: str, config: dict
) -> tuple[bool, str]:
    """Run a test query against the collection.

    Args:
        client: ChromaDB PersistentClient instance.
        collection_name: Name of the collection to query.
        config: Parsed eval_config.yaml dict (for embedding model name).

    Returns:
        (ok, error_string_or_empty)
    """
    try:
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        model_name = config["retrieval"]["embedding_model"]
        ef = SentenceTransformerEmbeddingFunction(model_name=model_name)

        query_collection = client.get_collection(
            name=collection_name, embedding_function=ef
        )

        n = min(3, query_collection.count())
        if n == 0:
            return (False, "collection is empty, cannot query")

        results = query_collection.query(
            query_texts=["test query"],
            n_results=n,
            include=["documents", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        if not docs:
            return (False, "test query returned no results")

        return (True, "")

    except Exception as e:
        return (False, f"query failed: {e}")


def _check_retriever_validate(
    collection: chromadb.Collection, config: dict
) -> tuple[bool, str]:
    """Call the existing validate_index() from cumrag.retriever.

    Returns:
        (ok, error_string_or_empty)
    """
    try:
        retriever_validate_index(collection, config)
        return (True, "")
    except ValueError as e:
        return (False, str(e))
    except Exception as e:
        return (False, f"unexpected error: {e}")


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------


def audit_index(
    index_dir: str = "index",
    n_samples: int = 5,
    verbose: bool = False,
) -> list[CollectionReport]:
    """Run full integrity audit on all ChromaDB collections.

    Args:
        index_dir: Path to ChromaDB persistence directory.
                   Relative paths resolve from project root.
        n_samples: Number of random entries to sample per collection.
        verbose: If True, log detailed progress.

    Returns:
        List of CollectionReport objects, one per collection.
    """
    index_path = Path(index_dir)
    if not index_path.is_absolute():
        index_path = _PROJECT_ROOT / index_path

    if not index_path.exists():
        logger.error("Index directory not found: %s", index_path)
        return []

    config = load_config("eval_config")
    client = chromadb.PersistentClient(path=str(index_path))
    names = _list_collection_names(client)

    if not names:
        logger.error("No collections found in %s", index_path)
        return []

    logger.info("Found %d collections in %s", len(names), index_path)

    reports: list[CollectionReport] = []

    for name in names:
        report = CollectionReport(name)
        collection = client.get_collection(name)
        report.count = collection.count()

        if verbose:
            logger.info("Validating '%s' (%d entries)...", name, report.count)

        # 1. Metadata validation
        report.metadata_ok, report.metadata_errors = _check_metadata(
            collection, config
        )

        # 2. Sample entry validation
        report.sample_ok, report.sample_errors = _check_samples(
            collection, n_samples
        )

        # 3. Test query
        report.query_ok, report.query_error = _check_query(client, name, config)

        # 4. Retriever validate_index()
        report.retriever_validate_ok, report.retriever_validate_error = (
            _check_retriever_validate(collection, config)
        )

        reports.append(report)

    return reports


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

_CHECK_PASS = "PASS"
_CHECK_FAIL = "FAIL"


def _status(ok: bool) -> str:
    return _CHECK_PASS if ok else _CHECK_FAIL


def print_report(reports: list[CollectionReport]) -> None:
    """Print validation results in a clear table format."""
    if not reports:
        print("No collections found to validate.")
        return

    # Header
    col_w = max(len(r.name) for r in reports)
    col_w = max(col_w, len("Collection"))

    header = (
        f"{'Collection':<{col_w}}  "
        f"{'Count':>7}  "
        f"{'Metadata':>8}  "
        f"{'Samples':>7}  "
        f"{'Query':>5}  "
        f"{'Retriever':>9}  "
        f"{'Overall':>7}"
    )
    sep = "=" * len(header)

    print()
    print(sep)
    print("CUMRAG ChromaDB Index Validation Report")
    print(sep)
    print(header)
    print("-" * len(header))

    for r in reports:
        row = (
            f"{r.name:<{col_w}}  "
            f"{r.count:>7,}  "
            f"{_status(r.metadata_ok):>8}  "
            f"{_status(r.sample_ok):>7}  "
            f"{_status(r.query_ok):>5}  "
            f"{_status(r.retriever_validate_ok):>9}  "
            f"{_status(r.passed):>7}"
        )
        print(row)

    print(sep)

    # Summary
    passed = sum(1 for r in reports if r.passed)
    total = len(reports)
    print(f"\nResult: {passed}/{total} collections passed all checks.")

    # Detail section for failures
    failures = [r for r in reports if not r.passed]
    if failures:
        print("\n--- Failure Details ---")
        for r in failures:
            print(f"\n  {r.name}:")
            if not r.metadata_ok:
                for err in r.metadata_errors:
                    print(f"    [Metadata] {err}")
            if not r.sample_ok:
                for err in r.sample_errors:
                    print(f"    [Sample]   {err}")
            if not r.query_ok:
                print(f"    [Query]    {r.query_error}")
            if not r.retriever_validate_ok:
                print(f"    [Retriever] {r.retriever_validate_error}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate ChromaDB index integrity for the CUMRAG eval harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --index-dir index --samples 10\n"
            "  %(prog)s --verbose\n"
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="index",
        help="Path to ChromaDB persistence directory (default: index/).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of random entries to sample per collection (default: 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Returns:
        0 if all collections pass, 1 if any fail.
    """
    args = parse_args(argv)

    import logging as _logging

    level = _logging.DEBUG if args.verbose else _logging.INFO
    setup_logging(level=level)

    reports = audit_index(
        index_dir=args.index_dir,
        n_samples=args.samples,
        verbose=args.verbose,
    )

    print_report(reports)

    if not reports:
        logger.error("No collections found — validation failed.")
        return 1

    all_passed = all(r.passed for r in reports)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
