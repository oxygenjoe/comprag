#!/usr/bin/env python3
"""Build ChromaDB vector index from eval dataset documents.

Chunks documents from downloaded eval datasets, embeds with
all-MiniLM-L6-v2 (384-dim), and persists to ChromaDB in index/.

Uses content-addressed collection names that encode indexing parameters
(dataset, embedding model, chunk size, overlap) via SHA256 hash prefix,
e.g. ``comprag_rgb_300w_a3f7c2d1``. This prevents stale index reuse when
any parameter changes.

The retrieval pipeline is CONSTANT across all eval runs. Same embedding
model, same chunking, same index. Only the generator model varies.

Usage (CLI):
    python scripts/build_index.py --dataset rgb --chunk-size 300 --overlap 64
    python scripts/build_index.py --dataset rgb --subset noise_robustness
    python scripts/build_index.py --all
    python scripts/build_index.py --all --force  # rebuild even if exists

Usage (module):
    from scripts.build_index import build_rgb_index, build_halueval_index, build_nq_index
    build_rgb_index("noise_robustness", chunk_size=300, overlap=64)
    build_halueval_index(chunk_size=300, overlap=64)
    build_nq_index(chunk_size=300, overlap=64)
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

from comprag.utils import get_logger, load_config, make_collection_name, setup_logging, timer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS_DIR = _PROJECT_ROOT / "datasets"
INDEX_DIR = _PROJECT_ROOT / "index"

EMBEDDING_DIM = 384
COMPRAG_VERSION = "2.0"


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

# RGB subsets — each gets its own ChromaDB collection
RGB_SUBSETS = (
    "noise_robustness",
    "negative_rejection",
    "information_integration",
    "counterfactual_robustness",
)

logger = get_logger("comprag.build_index")


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
    chunk_size, chunk_overlap), e.g. ``comprag_rgb_300w_a3f7c2d1``.

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
            "comprag_version": COMPRAG_VERSION,
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


def build_rgb_index(
    subset: str,
    chunk_size: int = 300,
    overlap: int = 64,
    force: bool = False,
    persist_dir: Optional[Path] = None,
    batch_size: int = 256,
) -> dict:
    """Build a ChromaDB index for a single RGB subset from normalized JSONL.

    RGB tests retrieval ROBUSTNESS, not retrieval quality. Each question
    comes with its own passages that must be indexed so the retriever
    finds them. This function reads the normalized JSONL produced by
    normalize_datasets.py and indexes each sample's ``metadata.original_passages``
    into a per-subset ChromaDB collection.

    Document IDs link back to the sample: ``{sample_id}_passage_{i}``.

    Args:
        subset: RGB subset name, one of: noise_robustness, negative_rejection,
                information_integration, counterfactual_robustness.
        chunk_size: Whitespace words per chunk (default 300).
        overlap: Overlapping words between chunks (default 64).
        force: If True, delete and rebuild existing collection.
        persist_dir: Override index directory (default: index/).
        batch_size: Number of chunks to embed and insert per batch.

    Returns:
        Dict with indexing stats.

    Raises:
        FileNotFoundError: If normalized JSONL file does not exist.
        ValueError: If no passages found to index.
    """
    if subset not in RGB_SUBSETS:
        raise ValueError(
            f"Invalid RGB subset '{subset}'. Must be one of: {', '.join(RGB_SUBSETS)}"
        )

    embedding_model = _get_embedding_model_name()

    # RGB subsets get per-subset collection names:
    #   dataset key = "rgb_{subset}" -> comprag_rgb_noise_robustness_300w_{hash}
    dataset_key = f"rgb_{subset}"
    collection_name = make_collection_name(
        dataset=dataset_key,
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
            "dataset": "rgb",
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
            pass

    # Locate normalized JSONL
    normalized_path = DATASETS_DIR / "rgb" / "normalized" / f"{subset}.jsonl"
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"Normalized RGB file not found: {normalized_path}\n"
            f"Run 'python scripts/normalize_datasets.py --datasets rgb' first."
        )

    with timer(f"RGB {subset} indexing", logger=logger) as t:
        # Step 1: Load normalized samples and extract passages
        logger.info("Loading normalized RGB samples from %s", normalized_path)
        samples = _load_jsonl(normalized_path)
        logger.info("Loaded %d samples for RGB/%s", len(samples), subset)

        # Step 2: Extract passages from metadata.original_passages and chunk them
        all_chunks = []
        num_passages = 0

        for sample in samples:
            sample_id = sample["sample_id"]
            metadata = sample.get("metadata", {})
            passages = metadata.get("original_passages", [])

            for passage_idx, passage in enumerate(passages):
                # Passages can be strings or dicts with a "text" field
                if isinstance(passage, str):
                    passage_text = passage.strip()
                elif isinstance(passage, dict):
                    passage_text = passage.get("text", passage.get("passage", "")).strip()
                else:
                    continue

                if not passage_text:
                    continue

                num_passages += 1

                # Chunk the passage (most RGB passages fit in one chunk,
                # but chunk for consistency with the pipeline)
                text_chunks = chunk_text(passage_text, chunk_size=chunk_size, overlap=overlap)

                for chunk_idx, chunk_str in enumerate(text_chunks):
                    chunk_id = f"{sample_id}_passage_{passage_idx}"
                    if len(text_chunks) > 1:
                        chunk_id = f"{chunk_id}_chunk_{chunk_idx}"

                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_str,
                        "sample_id": sample_id,
                        "passage_index": passage_idx,
                        "chunk_index": chunk_idx,
                        "dataset": "rgb",
                        "subset": subset,
                    })

        if not all_chunks:
            raise ValueError(
                f"No passages found to index for RGB/{subset}. "
                f"Check that normalized JSONL has metadata.original_passages."
            )

        logger.info(
            "RGB/%s: %d passages -> %d chunks from %d samples",
            subset,
            num_passages,
            len(all_chunks),
            len(samples),
        )

        # Step 3: Create collection and insert
        embedding_fn = get_embedding_function()

        collection_metadata = {
            "embedding_model": embedding_model,
            "embedding_dim": EMBEDDING_DIM,
            "chunk_size_words": chunk_size,
            "chunk_overlap_words": overlap,
            "dataset": "rgb",
            "subset": subset,
            "index_type": "rgb_per_question",
            "num_samples": len(samples),
            "num_passages": num_passages,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "comprag_version": COMPRAG_VERSION,
        }

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata=collection_metadata,
        )

        logger.info(
            "Embedding and inserting %d RGB chunks (batch_size=%d) ...",
            len(all_chunks),
            batch_size,
        )

        total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(total_batches),
            desc=f"Embedding RGB/{subset}",
            unit="batch",
        ):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(all_chunks))
            batch = all_chunks[start:end]

            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [
                {
                    "sample_id": c["sample_id"],
                    "passage_index": c["passage_index"],
                    "chunk_index": c["chunk_index"],
                    "dataset": c["dataset"],
                    "subset": c["subset"],
                }
                for c in batch
            ]

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )

    stats = {
        "dataset": "rgb",
        "subset": subset,
        "collection_name": collection_name,
        "num_samples": len(samples),
        "num_passages": num_passages,
        "num_chunks": len(all_chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": embedding_model,
        "skipped": False,
        "elapsed_seconds": round(t.elapsed, 2),
    }

    logger.info(
        "RGB index built for '%s': %d chunks in collection '%s' (%.1fs)",
        subset,
        len(all_chunks),
        collection_name,
        t.elapsed,
    )

    return stats


def build_nq_index(
    chunk_size: int = 300,
    overlap: int = 64,
    force: bool = False,
    persist_dir: Optional[Path] = None,
    batch_size: int = 256,
) -> dict:
    """Build a ChromaDB index for Natural Questions from normalized JSONL.

    Reads ``datasets/nq/normalized/test.jsonl`` (produced by
    normalize_datasets.py) and indexes each sample's ground_truth and
    all_answers as a synthetic answer document. Document IDs link back
    to the sample via ``sample_id``.

    In production, NQ retrieval will use the full Wikipedia corpus on
    Optane. This index uses ``index_type: nq_normalized_test`` so the
    eval pipeline can distinguish it from a Wikipedia-backed index and
    swap it out later without changing the rest of the pipeline.

    Follows the same pattern as ``build_halueval_index()``:
    read normalized JSONL -> extract text fields -> chunk -> embed -> persist.

    Args:
        chunk_size: Whitespace words per chunk (default 300).
        overlap: Overlapping words between chunks (default 64).
        force: If True, delete and rebuild existing collection.
        persist_dir: Override index directory (default: index/).
        batch_size: Number of chunks to embed and insert per batch.

    Returns:
        Dict with indexing stats.

    Raises:
        FileNotFoundError: If normalized JSONL file does not exist.
        ValueError: If no documents could be created.
    """
    normalized_path = DATASETS_DIR / "nq" / "normalized" / "test.jsonl"
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"Normalized NQ file not found: {normalized_path}\n"
            f"Run 'python scripts/normalize_datasets.py --datasets nq' first."
        )

    embedding_model = _get_embedding_model_name()
    collection_name = make_collection_name(
        dataset="nq",
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    client = get_chroma_client(persist_dir)

    if not force and collection_exists(client, collection_name):
        count = client.get_collection(collection_name).count()
        logger.info(
            "Collection '%s' already exists with %d entries. Use --force to rebuild.",
            collection_name, count,
        )
        return {
            "dataset": "nq",
            "subset": "test",
            "collection_name": collection_name,
            "num_chunks": count,
            "skipped": True,
            "elapsed_seconds": 0.0,
        }

    if force:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s' for rebuild.", collection_name)
        except Exception:
            pass

    with timer("NQ indexing", logger=logger) as t:
        # Step 1: Load normalized NQ samples
        logger.info("Loading normalized NQ samples from %s", normalized_path)
        samples = _load_jsonl(normalized_path)
        logger.info("Loaded %d NQ samples", len(samples))

        # Step 2: Build answer documents from ground_truth + all_answers
        all_chunks = []
        skipped_no_answer = 0

        for sample in samples:
            sample_id = sample["sample_id"]
            ground_truth = sample.get("ground_truth", "")
            metadata = sample.get("metadata", {})
            all_answers = metadata.get("all_answers", [])

            if not ground_truth:
                skipped_no_answer += 1
                continue

            # Combine ground_truth with alternative answers as document text
            answer_parts = [ground_truth]
            answer_parts.extend(
                a for a in all_answers if a and a != ground_truth
            )
            doc_text = " ".join(answer_parts)

            # Chunk the document (most NQ answers are short enough for one chunk)
            text_chunks = chunk_text(
                doc_text, chunk_size=chunk_size, overlap=overlap
            )

            for chunk_idx, chunk_str in enumerate(text_chunks):
                chunk_id = f"{sample_id}_answer"
                if len(text_chunks) > 1:
                    chunk_id = f"{chunk_id}_chunk_{chunk_idx}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_str,
                    "sample_id": sample_id,
                    "chunk_index": chunk_idx,
                    "dataset": "nq",
                    "subset": "test",
                })

        if skipped_no_answer:
            logger.warning(
                "NQ: %d samples had no ground_truth field", skipped_no_answer
            )

        if not all_chunks:
            raise ValueError(
                "No documents could be created from NQ test data. "
                "Check that normalized JSONL has ground_truth fields."
            )

        logger.info(
            "NQ: %d answer chunks from %d samples",
            len(all_chunks),
            len(samples),
        )

        # Step 3: Create collection and insert
        embedding_fn = get_embedding_function()

        collection_metadata = {
            "embedding_model": embedding_model,
            "embedding_dim": EMBEDDING_DIM,
            "chunk_size_words": chunk_size,
            "chunk_overlap_words": overlap,
            "dataset": "nq",
            "subset": "test",
            "index_type": "nq_normalized_test",
            "num_samples": len(samples),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "comprag_version": COMPRAG_VERSION,
        }

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata=collection_metadata,
        )

        logger.info(
            "Embedding and inserting %d NQ chunks (batch_size=%d) ...",
            len(all_chunks),
            batch_size,
        )

        total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(total_batches),
            desc="Embedding NQ",
            unit="batch",
        ):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(all_chunks))
            batch = all_chunks[start:end]

            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [
                {
                    "sample_id": c["sample_id"],
                    "chunk_index": c["chunk_index"],
                    "dataset": c["dataset"],
                    "subset": c["subset"],
                }
                for c in batch
            ]

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )

    stats = {
        "dataset": "nq",
        "subset": "test",
        "collection_name": collection_name,
        "num_samples": len(samples),
        "num_chunks": len(all_chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": embedding_model,
        "index_type": "nq_normalized_test",
        "skipped": False,
        "elapsed_seconds": round(t.elapsed, 2),
    }

    logger.info(
        "NQ index built: %d chunks in collection '%s' (%.1fs)",
        len(all_chunks),
        collection_name,
        t.elapsed,
    )

    return stats


def build_halueval_index(
    chunk_size: int = 300,
    overlap: int = 64,
    force: bool = False,
    persist_dir: Optional[Path] = None,
    batch_size: int = 256,
) -> dict:
    """Build a ChromaDB index for HaluEval from normalized JSONL.

    HaluEval provides a ``knowledge`` field per sample. This function
    indexes each sample's knowledge text into a single ``comprag_halueval_*``
    collection, with document IDs that link back to the sample.

    Args:
        chunk_size: Whitespace words per chunk (default 300).
        overlap: Overlapping words between chunks (default 64).
        force: If True, delete and rebuild existing collection.
        persist_dir: Override index directory (default: index/).
        batch_size: Number of chunks to embed and insert per batch.

    Returns:
        Dict with indexing stats.

    Raises:
        FileNotFoundError: If normalized JSONL file does not exist.
        ValueError: If no knowledge fields found to index.
    """
    embedding_model = _get_embedding_model_name()

    collection_name = make_collection_name(
        dataset="halueval",
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
            "dataset": "halueval",
            "subset": "qa",
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
            pass

    # Locate normalized JSONL
    normalized_path = DATASETS_DIR / "halueval" / "normalized" / "qa.jsonl"
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"Normalized HaluEval file not found: {normalized_path}\n"
            f"Run 'python scripts/normalize_datasets.py --datasets halueval' first."
        )

    with timer("HaluEval indexing", logger=logger) as t:
        # Step 1: Load normalized samples and extract knowledge fields
        logger.info("Loading normalized HaluEval samples from %s", normalized_path)
        samples = _load_jsonl(normalized_path)
        logger.info("Loaded %d HaluEval samples", len(samples))

        # Step 2: Extract knowledge text and chunk
        all_chunks = []
        skipped_no_knowledge = 0

        for sample in samples:
            sample_id = sample["sample_id"]
            metadata = sample.get("metadata", {})
            knowledge = metadata.get("knowledge", "")

            if not knowledge or not knowledge.strip():
                skipped_no_knowledge += 1
                continue

            knowledge_text = knowledge.strip()
            text_chunks = chunk_text(knowledge_text, chunk_size=chunk_size, overlap=overlap)

            for chunk_idx, chunk_str in enumerate(text_chunks):
                chunk_id = f"{sample_id}_knowledge"
                if len(text_chunks) > 1:
                    chunk_id = f"{chunk_id}_chunk_{chunk_idx}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_str,
                    "sample_id": sample_id,
                    "chunk_index": chunk_idx,
                    "dataset": "halueval",
                    "subset": "qa",
                })

        if skipped_no_knowledge:
            logger.warning(
                "HaluEval: %d samples had no knowledge field", skipped_no_knowledge
            )

        if not all_chunks:
            raise ValueError(
                "No knowledge fields found to index for HaluEval. "
                "Check that normalized JSONL has metadata.knowledge."
            )

        logger.info(
            "HaluEval: %d knowledge chunks from %d samples",
            len(all_chunks),
            len(samples),
        )

        # Step 3: Create collection and insert
        embedding_fn = get_embedding_function()

        collection_metadata = {
            "embedding_model": embedding_model,
            "embedding_dim": EMBEDDING_DIM,
            "chunk_size_words": chunk_size,
            "chunk_overlap_words": overlap,
            "dataset": "halueval",
            "subset": "qa",
            "index_type": "halueval_knowledge",
            "num_samples": len(samples),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "comprag_version": COMPRAG_VERSION,
        }

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata=collection_metadata,
        )

        logger.info(
            "Embedding and inserting %d HaluEval chunks (batch_size=%d) ...",
            len(all_chunks),
            batch_size,
        )

        total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(total_batches),
            desc="Embedding HaluEval",
            unit="batch",
        ):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(all_chunks))
            batch = all_chunks[start:end]

            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [
                {
                    "sample_id": c["sample_id"],
                    "chunk_index": c["chunk_index"],
                    "dataset": c["dataset"],
                    "subset": c["subset"],
                }
                for c in batch
            ]

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )

    stats = {
        "dataset": "halueval",
        "subset": "qa",
        "collection_name": collection_name,
        "num_samples": len(samples),
        "num_chunks": len(all_chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": embedding_model,
        "skipped": False,
        "elapsed_seconds": round(t.elapsed, 2),
    }

    logger.info(
        "HaluEval index built: %d chunks in collection '%s' (%.1fs)",
        len(all_chunks),
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

    Dispatches to dataset-specific indexing logic:
    - RGB: builds one collection per subset via build_rgb_index()
    - HaluEval: indexes knowledge fields via build_halueval_index()
    - NQ: builds stub corpus from ground_truth answers for integration testing

    Skips datasets that don't have downloaded/normalized data yet.

    Args:
        chunk_size: Tokens per chunk.
        overlap: Overlap tokens.
        force: Rebuild existing collections.
        persist_dir: Override index directory.
        batch_size: Embedding batch size.

    Returns:
        List of stats dicts (one per collection built).
    """
    results = []

    # --- RGB: per-subset indexing ---
    for subset in RGB_SUBSETS:
        try:
            stats = build_rgb_index(
                subset=subset,
                chunk_size=chunk_size,
                overlap=overlap,
                force=force,
                persist_dir=persist_dir,
                batch_size=batch_size,
            )
            results.append(stats)
        except FileNotFoundError as e:
            logger.warning("Skipping RGB/%s: %s", subset, e)
        except ValueError as e:
            logger.warning("Skipping RGB/%s: %s", subset, e)
        except Exception as e:
            logger.error("Failed to index RGB/%s: %s", subset, e, exc_info=True)

    # --- NQ: answer corpus from normalized JSONL ground_truth + all_answers ---
    try:
        stats = build_nq_index(
            chunk_size=chunk_size,
            overlap=overlap,
            force=force,
            persist_dir=persist_dir,
            batch_size=batch_size,
        )
        results.append(stats)
    except Exception as e:
        logger.warning("Skipping NQ: %s", e)

    # --- HaluEval: knowledge field indexing ---
    try:
        stats = build_halueval_index(
            chunk_size=chunk_size,
            overlap=overlap,
            force=force,
            persist_dir=persist_dir,
            batch_size=batch_size,
        )
        results.append(stats)
    except FileNotFoundError as e:
        logger.warning("Skipping HaluEval: %s", e)
    except ValueError as e:
        logger.warning("Skipping HaluEval: %s", e)
    except Exception as e:
        logger.error("Failed to index HaluEval: %s", e, exc_info=True)

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
        elif args.dataset == "rgb":
            # RGB: per-subset indexing from normalized JSONL
            if args.subset:
                # Single subset
                result = build_rgb_index(
                    subset=args.subset,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                    force=args.force,
                    persist_dir=persist_dir,
                    batch_size=args.batch_size,
                )
                results = [result]
            else:
                # All RGB subsets
                results = []
                for subset in RGB_SUBSETS:
                    result = build_rgb_index(
                        subset=subset,
                        chunk_size=args.chunk_size,
                        overlap=args.overlap,
                        force=args.force,
                        persist_dir=persist_dir,
                        batch_size=args.batch_size,
                    )
                    results.append(result)
        elif args.dataset == "halueval":
            # HaluEval: knowledge field indexing from normalized JSONL
            result = build_halueval_index(
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                force=args.force,
                persist_dir=persist_dir,
                batch_size=args.batch_size,
            )
            results = [result]
        elif args.dataset == "nq":
            result = build_nq_index(
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                force=args.force,
                persist_dir=persist_dir,
                batch_size=args.batch_size,
            )
            results = [result]
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
            label = r["dataset"]
            if r.get("subset"):
                label = f"{label}/{r['subset']}"
            print(
                f"  {label:<35} {status:<20} "
                f"chunks={r['num_chunks']:<8} "
                f"time={r.get('elapsed_seconds', 0):.1f}s"
            )

        if not results:
            print("  (no indexes built)")

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
