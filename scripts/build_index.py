#!/usr/bin/env python3
"""Build ChromaDB vector index from eval dataset documents.

Chunks documents from downloaded eval datasets, embeds with
all-MiniLM-L6-v2 (384-dim), and persists to ChromaDB in index/.

Uses content-addressed collection names that encode indexing parameters
(dataset, embedding model, chunk size, overlap) via SHA256 hash prefix,
e.g. ``cumrag_rgb_300w_a3f7c2d1``. This prevents stale index reuse when
any parameter changes.

The retrieval pipeline is CONSTANT across all eval runs. Same embedding
model, same chunking, same index. Only the generator model varies.

Usage (CLI):
    python scripts/build_index.py --dataset rgb --chunk-size 300 --overlap 64
    python scripts/build_index.py --dataset rgb --subset noise_robustness
    python scripts/build_index.py --all
    python scripts/build_index.py --all --force  # rebuild even if exists

Usage (module):
    from scripts.build_index import build_index, chunk_documents
    build_index("rgb", chunk_size=300, overlap=64, subset="noise_robustness")
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Ensure the project root is importable
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cumrag.utils import get_logger, load_config, make_collection_name, setup_logging, timer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS_DIR = _PROJECT_ROOT / "datasets"
INDEX_DIR = _PROJECT_ROOT / "index"

EMBEDDING_DIM = 384
CUMRAG_VERSION = "2.0"


def _get_embedding_model_name() -> str:
    """Read embedding model name from eval_config.yaml (single source of truth)."""
    try:
        config = load_config("eval_config")
        return config["retrieval"]["embedding_model"]
    except (FileNotFoundError, KeyError):
        logger.warning(
            "Could not read retrieval.embedding_model from eval_config.yaml, "
            "falling back to 'all-MiniLM-L6-v2'"
        )
        return "all-MiniLM-L6-v2"

VALID_DATASETS = ("rgb", "nq", "halueval")

logger = get_logger("cumrag.build_index")


# ---------------------------------------------------------------------------
# Tokenizer — whitespace-based with consistent behavior
# ---------------------------------------------------------------------------


class WhitespaceTokenizer:
    """Simple whitespace tokenizer for deterministic chunking.

    Splits on whitespace boundaries. One "token" = one whitespace-delimited
    word. This keeps chunking fast, dependency-free, and deterministic.
    Sentence-transformers models handle subword tokenization internally,
    so we only need approximate token counts for chunk sizing.
    """

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split text into whitespace-delimited tokens."""
        return text.split()

    @staticmethod
    def detokenize(tokens: list[str]) -> str:
        """Join tokens back with single spaces."""
        return " ".join(tokens)


_tokenizer = WhitespaceTokenizer()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = 300,  # ~300 whitespace words ≈ 400-500 BPE tokens, safe for top_k=5 within 4096 ctx
    overlap: int = 64,
) -> list[str]:
    """Split text into overlapping chunks by token count.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum tokens per chunk.
        overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of chunk strings. Empty list if text is empty/whitespace.
    """
    tokens = _tokenizer.tokenize(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    stride = max(1, chunk_size - overlap)

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = _tokenizer.detokenize(chunk_tokens)
        if chunk_str.strip():
            chunks.append(chunk_str)
        start += stride

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 300,  # ~300 whitespace words ≈ 400-500 BPE tokens, safe for top_k=5 within 4096 ctx
    overlap: int = 64,
) -> list[dict]:
    """Chunk a list of documents into retrieval-sized pieces.

    Each document dict must have at least a "text" field. Additional fields
    (source, dataset, doc_id, etc.) are preserved as metadata on each chunk.

    Args:
        documents: List of dicts with "text" and optional metadata.
        chunk_size: Maximum tokens per chunk.
        overlap: Overlapping tokens between consecutive chunks.

    Returns:
        List of chunk dicts with keys:
            - chunk_id: deterministic ID (hash of content)
            - text: chunk text
            - doc_id: source document identifier
            - chunk_index: 0-based index within the source document
            - dataset: dataset name
            - source: source document identifier
    """
    all_chunks = []

    for doc in tqdm(documents, desc="Chunking documents", unit="doc"):
        text = doc.get("text", "")
        if not text or not text.strip():
            continue

        doc_id = doc.get("doc_id", doc.get("source", "unknown"))
        dataset = doc.get("dataset", "unknown")

        text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk_str in enumerate(text_chunks):
            # Deterministic chunk ID from content for idempotency
            content_hash = hashlib.md5(
                f"{dataset}:{doc_id}:{idx}:{chunk_str}".encode("utf-8")
            ).hexdigest()[:12]

            chunk_record = {
                "chunk_id": f"{dataset}_{doc_id}_{idx}_{content_hash}",
                "text": chunk_str,
                "doc_id": str(doc_id),
                "chunk_index": idx,
                "dataset": dataset,
                "source": str(doc_id),
            }
            all_chunks.append(chunk_record)

    return all_chunks


# ---------------------------------------------------------------------------
# Dataset loaders — read normalized dataset files from datasets/
# ---------------------------------------------------------------------------


def _load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file, returning a list of dicts."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _load_json(filepath: Path) -> list[dict]:
    """Load a JSON file (list of dicts or single dict)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def _extract_documents_from_dataset(records: list[dict], dataset_name: str) -> list[dict]:
    """Extract indexable documents from normalized dataset records.

    Datasets typically have question/answer/context fields. For building
    the retrieval index, we extract the context/passage text. If a record
    has no context, we skip it (nothing to index).

    Also handles raw document formats where the text lives in different
    fields depending on the dataset source.
    """
    documents = []
    seen_texts = set()  # deduplicate identical passages

    for i, record in enumerate(records):
        # Try multiple field names for the document/context text
        text = None
        for field in ("context", "passage", "text", "document", "content", "paragraph"):
            val = record.get(field)
            if val and isinstance(val, str) and val.strip():
                text = val.strip()
                break

        # Some datasets have contexts as a list of passages
        if text is None:
            for field in ("contexts", "passages", "documents", "paragraphs"):
                val = record.get(field)
                if val and isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and item.strip():
                            item_text = item.strip()
                            if item_text not in seen_texts:
                                seen_texts.add(item_text)
                                doc_id = record.get("id", record.get("question_id", f"doc_{i}"))
                                documents.append({
                                    "text": item_text,
                                    "doc_id": str(doc_id),
                                    "dataset": dataset_name,
                                })
                    continue

        if text is None:
            continue

        # Deduplicate
        if text in seen_texts:
            continue
        seen_texts.add(text)

        doc_id = record.get("id", record.get("question_id", record.get("source", f"doc_{i}")))
        documents.append({
            "text": text,
            "doc_id": str(doc_id),
            "dataset": dataset_name,
        })

    return documents


def load_dataset_documents(dataset_name: str) -> list[dict]:
    """Load and extract indexable documents from a dataset directory.

    Scans the dataset directory for JSONL and JSON files, loads them,
    and extracts context/passage text for indexing.

    Args:
        dataset_name: One of 'rgb', 'nq', 'halueval'.

    Returns:
        List of document dicts with 'text', 'doc_id', 'dataset' keys.

    Raises:
        FileNotFoundError: If dataset directory doesn't exist.
        ValueError: If no indexable documents found.
    """
    dataset_dir = DATASETS_DIR / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Run 'python scripts/download_datasets.py --dataset {dataset_name}' first."
        )

    # Collect all JSONL and JSON files
    data_files = sorted(
        list(dataset_dir.glob("**/*.jsonl"))
        + list(dataset_dir.glob("**/*.json"))
    )

    if not data_files:
        raise ValueError(
            f"No JSONL or JSON files found in {dataset_dir}. "
            f"Is the dataset downloaded?"
        )

    all_documents = []

    for filepath in data_files:
        logger.info("Loading %s", filepath.relative_to(_PROJECT_ROOT))
        try:
            if filepath.suffix == ".jsonl":
                records = _load_jsonl(filepath)
            else:
                records = _load_json(filepath)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Skipping %s: %s", filepath.name, e)
            continue

        docs = _extract_documents_from_dataset(records, dataset_name)
        all_documents.extend(docs)
        logger.info("  Extracted %d documents from %s", len(docs), filepath.name)

    if not all_documents:
        raise ValueError(
            f"No indexable documents found in {dataset_dir}. "
            f"Dataset files exist but contain no context/passage text."
        )

    logger.info(
        "Total documents for '%s': %d (deduplicated)",
        dataset_name,
        len(all_documents),
    )
    return all_documents


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def get_embedding_function():
    """Create a ChromaDB-compatible embedding function using sentence-transformers.

    Returns:
        A chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
        configured with all-MiniLM-L6-v2.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(
        model_name=_get_embedding_model_name(),
    )


# ---------------------------------------------------------------------------
# ChromaDB index construction
# ---------------------------------------------------------------------------


def get_chroma_client(persist_dir: Optional[Path] = None):
    """Create a persistent ChromaDB client.

    Args:
        persist_dir: Directory for ChromaDB persistence. Defaults to index/.

    Returns:
        chromadb.PersistentClient instance.
    """
    import chromadb

    persist_dir = persist_dir or INDEX_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(persist_dir))


def collection_exists(client, collection_name: str) -> bool:
    """Check if a ChromaDB collection already exists and has data.

    Args:
        client: ChromaDB PersistentClient.
        collection_name: Name of the collection to check.

    Returns:
        True if collection exists and contains at least one record.
    """
    try:
        existing = client.list_collections()
        # chromadb >= 0.4 returns list of Collection objects or strings
        names = []
        for c in existing:
            if isinstance(c, str):
                names.append(c)
            elif hasattr(c, "name"):
                names.append(c.name)
        if collection_name not in names:
            return False
        # Check if collection has data
        col = client.get_collection(collection_name)
        return col.count() > 0
    except Exception:
        return False


def build_index(
    dataset_name: str,
    chunk_size: int = 300,  # ~300 whitespace words ≈ 400-500 BPE tokens, safe for top_k=5 within 4096 ctx
    overlap: int = 64,
    subset: Optional[str] = None,
    force: bool = False,
    persist_dir: Optional[Path] = None,
    batch_size: int = 256,
) -> dict:
    """Build a ChromaDB index for a single dataset.

    Loads documents from the dataset directory, chunks them, embeds with
    all-MiniLM-L6-v2, and persists to ChromaDB. Each dataset gets its
    own content-addressed collection with versioned naming.

    Collection names are deterministic based on (dataset, embedding_model,
    chunk_size, chunk_overlap), e.g. ``cumrag_rgb_300w_a3f7c2d1``.

    Args:
        dataset_name: One of 'rgb', 'nq', 'halueval'.
        chunk_size: Whitespace words per chunk (default 300).
        overlap: Overlapping words between chunks (default 64).
        subset: Optional subset name for per-subset collections
                (e.g. "noise_robustness" for RGB). Stored in metadata.
        force: If True, delete and rebuild existing collection.
        persist_dir: Override index directory (default: index/).
        batch_size: Number of chunks to embed and insert per batch.

    Returns:
        Dict with indexing stats: dataset, subset, num_documents, num_chunks,
        collection_name, elapsed_seconds.

    Raises:
        FileNotFoundError: If dataset not downloaded yet.
        ValueError: If no indexable documents found.
    """
    embedding_model = _get_embedding_model_name()

    # Content-addressed collection name from indexing parameters
    collection_name = make_collection_name(
        dataset=dataset_name,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    client = get_chroma_client(persist_dir)

    # Idempotency check
    if not force and collection_exists(client, collection_name):
        count = client.get_collection(collection_name).count()
        logger.info(
            "Collection '%s' already exists with %d entries. "
            "Use --force to rebuild.",
            collection_name,
            count,
        )
        return {
            "dataset": dataset_name,
            "subset": subset,
            "collection_name": collection_name,
            "num_chunks": count,
            "skipped": True,
            "elapsed_seconds": 0.0,
        }

    # Delete existing collection if force rebuild
    if force:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s' for rebuild.", collection_name)
        except Exception:
            pass  # Collection didn't exist, that's fine

    with timer("Total indexing", logger=logger) as t:
        # Step 1: Load documents
        with timer(f"Loading {dataset_name} documents", logger=logger):
            documents = load_dataset_documents(dataset_name)

        # Step 2: Chunk
        with timer(f"Chunking {len(documents)} documents", logger=logger):
            chunks = chunk_documents(
                documents,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        logger.info(
            "Produced %d chunks from %d documents (chunk_size=%d, overlap=%d)",
            len(chunks),
            len(documents),
            chunk_size,
            overlap,
        )

        if not chunks:
            raise ValueError(
                f"Chunking produced zero chunks for dataset '{dataset_name}'. "
                f"Check that documents have sufficient text content."
            )

        # Step 3: Create collection with full indexing metadata and embed + insert
        embedding_fn = get_embedding_function()

        collection_metadata = {
            "embedding_model": embedding_model,
            "embedding_dim": EMBEDDING_DIM,
            "chunk_size_words": chunk_size,
            "chunk_overlap_words": overlap,
            "dataset": dataset_name,
            "subset": subset or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cumrag_version": CUMRAG_VERSION,
        }

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata=collection_metadata,
        )

        logger.info(
            "Embedding and inserting %d chunks (batch_size=%d) ...",
            len(chunks),
            batch_size,
        )

        # Process in batches for memory efficiency and progress tracking
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(total_batches),
            desc=f"Embedding {dataset_name}",
            unit="batch",
        ):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(chunks))
            batch = chunks[start:end]

            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [
                {
                    "doc_id": c["doc_id"],
                    "chunk_index": c["chunk_index"],
                    "dataset": c["dataset"],
                    "source": c["source"],
                }
                for c in batch
            ]

            # ChromaDB handles embedding via the collection's embedding function
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )

    stats = {
        "dataset": dataset_name,
        "subset": subset,
        "collection_name": collection_name,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": embedding_model,
        "skipped": False,
        "elapsed_seconds": round(t.elapsed, 2),
    }

    logger.info(
        "Index built for '%s': %d chunks in collection '%s' (%.1fs)",
        dataset_name,
        len(chunks),
        collection_name,
        t.elapsed,
    )

    return stats


def build_all(
    chunk_size: int = 300,  # ~300 whitespace words ≈ 400-500 BPE tokens, safe for top_k=5 within 4096 ctx
    overlap: int = 64,
    force: bool = False,
    persist_dir: Optional[Path] = None,
    batch_size: int = 256,
) -> list[dict]:
    """Build ChromaDB indexes for all available datasets.

    Skips datasets that don't have downloaded data yet (logs a warning).

    Args:
        chunk_size: Tokens per chunk.
        overlap: Overlap tokens.
        force: Rebuild existing collections.
        persist_dir: Override index directory.
        batch_size: Embedding batch size.

    Returns:
        List of stats dicts (one per dataset attempted).
    """
    results = []

    for ds_name in VALID_DATASETS:
        try:
            stats = build_index(
                dataset_name=ds_name,
                chunk_size=chunk_size,
                overlap=overlap,
                force=force,
                persist_dir=persist_dir,
                batch_size=batch_size,
            )
            results.append(stats)
        except FileNotFoundError as e:
            logger.warning("Skipping '%s': %s", ds_name, e)
        except ValueError as e:
            logger.warning("Skipping '%s': %s", ds_name, e)
        except Exception as e:
            logger.error("Failed to index '%s': %s", ds_name, e, exc_info=True)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build ChromaDB vector index from eval dataset documents. "
            "Chunks, embeds (all-MiniLM-L6-v2), and persists to index/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --dataset rgb\n"
            "  %(prog)s --dataset rgb --subset noise_robustness\n"
            "  %(prog)s --dataset rgb --chunk-size 512 --overlap 64\n"
            "  %(prog)s --all\n"
            "  %(prog)s --all --force\n"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        choices=VALID_DATASETS,
        help="Build index for a specific dataset.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="all_datasets",
        help="Build indexes for all available datasets.",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional subset name for per-subset collections (e.g. 'noise_robustness' for RGB).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Maximum whitespace words per chunk (default: 300, ≈400-500 BPE tokens).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlapping tokens between chunks (default: 64).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if collection already exists.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help=f"Override index directory (default: {INDEX_DIR}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of chunks per embedding/insert batch (default: 256).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 on failure.
    """
    args = parse_args(argv)

    import logging as _logging

    level = _logging.DEBUG if args.verbose else _logging.INFO
    setup_logging(level=level)

    persist_dir = Path(args.index_dir) if args.index_dir else None

    try:
        if args.all_datasets:
            results = build_all(
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                force=args.force,
                persist_dir=persist_dir,
                batch_size=args.batch_size,
            )
        else:
            result = build_index(
                dataset_name=args.dataset,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                subset=args.subset,
                force=args.force,
                persist_dir=persist_dir,
                batch_size=args.batch_size,
            )
            results = [result]

        # Print summary
        print("\n" + "=" * 60)
        print("Indexing Summary")
        print("=" * 60)

        for r in results:
            status = "SKIPPED (exists)" if r.get("skipped") else "BUILT"
            print(
                f"  {r['dataset']:<12} {status:<20} "
                f"chunks={r['num_chunks']:<8} "
                f"time={r.get('elapsed_seconds', 0):.1f}s"
            )

        print("=" * 60)
        return 0

    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
