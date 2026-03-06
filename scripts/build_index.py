#!/usr/bin/env python3
"""Build ChromaDB index from normalized dataset files.

Reads normalized JSONL from datasets/<name>/normalized/, chunks text
with word-based sliding window, embeds with sentence-transformers,
and stores in persistent ChromaDB. One collection per dataset-subset
(e.g. "rgb_noise_robustness"). Idempotent unless --force is passed.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 300
DEFAULT_OVERLAP = 64


def load_config() -> dict:
    """Load retrieval config from eval.yaml if available."""
    config_path = PROJECT_ROOT / "config" / "eval.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("retrieval", {})
    except (ImportError, Exception) as e:
        logger.warning("Failed to load eval.yaml: %s", e)
        return {}


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def extract_passages(sample: dict) -> list[str]:
    """Extract indexable text from a normalized sample.

    Handles RGB (metadata.original_passages), HaluEval (metadata.knowledge),
    and gracefully returns empty for datasets without passages (NQ).
    """
    meta = sample.get("metadata", {})
    passages: list[str] = []
    original = meta.get("original_passages")
    if original and isinstance(original, list):
        passages.extend(p for p in original if isinstance(p, str) and p.strip())
    knowledge = meta.get("knowledge")
    if knowledge and isinstance(knowledge, str) and knowledge.strip():
        passages.append(knowledge)
    return passages


def discover_normalized_files(
    datasets_dir: Path, dataset_filter: Optional[str] = None,
) -> list[tuple[str, str, Path]]:
    """Find all normalized JSONL files, optionally filtered by dataset."""
    results: list[tuple[str, str, Path]] = []
    if not datasets_dir.exists():
        logger.error("Datasets directory does not exist: %s", datasets_dir)
        return results
    for dataset_dir in sorted(datasets_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_filter and dataset_dir.name != dataset_filter:
            continue
        norm_dir = dataset_dir / "normalized"
        if not norm_dir.exists():
            continue
        for jsonl_file in sorted(norm_dir.glob("*.jsonl")):
            results.append((dataset_dir.name, jsonl_file.stem, jsonl_file))
    return results


def load_and_chunk_file(
    file_path: Path, dataset: str, subset: str,
    chunk_size: int, overlap: int,
) -> tuple[list[str], list[str], list[dict]]:
    """Load a normalized JSONL file, extract passages, and chunk them."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    counter = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Bad JSON at %s:%d: %s", file_path, line_num, e)
                    continue
                sid = sample.get("sample_id", f"{dataset}_{subset}_{line_num}")
                for passage in extract_passages(sample):
                    for i, chunk in enumerate(chunk_text(passage, chunk_size, overlap)):
                        ids.append(f"{sid}_chunk_{counter}")
                        texts.append(chunk)
                        metadatas.append({
                            "sample_id": sid, "dataset": dataset,
                            "subset": subset, "chunk_index": i,
                            "query": sample.get("query", ""),
                        })
                        counter += 1
    except OSError as e:
        logger.error("Failed to read %s: %s", file_path, e)
        raise
    return ids, texts, metadatas


def build_collection(
    client: chromadb.ClientAPI,
    embedding_fn: SentenceTransformerEmbeddingFunction,
    name: str, ids: list[str], texts: list[str],
    metadatas: list[dict], batch_size: int = 500,
) -> None:
    """Create a ChromaDB collection and insert chunks in batches."""
    col = client.get_or_create_collection(name=name, embedding_function=embedding_fn)
    total = len(ids)
    logger.info("Inserting %d chunks into '%s'", total, name)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        col.add(ids=ids[start:end], documents=texts[start:end],
                metadatas=metadatas[start:end])
        logger.info("  Batch %d-%d / %d", start, end, total)
    logger.info("Collection '%s' complete: %d chunks", name, col.count())


def collection_exists(client: chromadb.ClientAPI, name: str) -> bool:
    """Check whether a collection already exists."""
    return name in [c.name for c in client.list_collections()]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Build ChromaDB index from normalized dataset files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  %(prog)s                         # index all datasets\n"
               "  %(prog)s --dataset rgb --force    # rebuild RGB only\n",
    )
    p.add_argument("--dataset", choices=["rgb", "nq", "halueval"], default=None,
                   help="Build index for a specific dataset (default: all)")
    p.add_argument("--index-dir", type=Path, default=None,
                   help=f"ChromaDB storage directory (default: {DEFAULT_INDEX_DIR})")
    p.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR,
                   help=f"Root datasets directory (default: {DATASETS_DIR})")
    p.add_argument("--force", action="store_true",
                   help="Rebuild existing collections")
    p.add_argument("--chunk-size", type=int, default=None,
                   help=f"Words per chunk (default: {DEFAULT_CHUNK_SIZE})")
    p.add_argument("--overlap", type=int, default=None,
                   help=f"Overlap words (default: {DEFAULT_OVERLAP})")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   default="INFO", help="Logging level (default: INFO)")
    return p.parse_args(argv)


def process_files(
    files: list[tuple[str, str, Path]],
    client: chromadb.ClientAPI,
    embedding_fn: SentenceTransformerEmbeddingFunction,
    chunk_size: int, overlap: int, force: bool,
) -> tuple[int, int]:
    """Process all discovered files, building or skipping collections."""
    built, skipped = 0, 0
    for dataset, subset, file_path in files:
        col_name = f"{dataset}_{subset}"
        if collection_exists(client, col_name):
            if force:
                logger.info("Deleting '%s' (--force)", col_name)
                client.delete_collection(col_name)
            else:
                logger.info("Skipping existing '%s'", col_name)
                skipped += 1
                continue
        logger.info("Processing %s/%s from %s", dataset, subset, file_path)
        ids, texts, metas = load_and_chunk_file(
            file_path, dataset, subset, chunk_size, overlap)
        if not ids:
            logger.warning("No chunks for %s/%s — no indexable passages", dataset, subset)
            skipped += 1
            continue
        build_collection(client, embedding_fn, col_name, ids, texts, metas)
        built += 1
    return built, skipped


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the index builder."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = load_config()
    chunk_size = args.chunk_size or config.get("chunk_size_words", DEFAULT_CHUNK_SIZE)
    overlap = args.overlap or config.get("overlap_words", DEFAULT_OVERLAP)
    embedding_model = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    index_dir = args.index_dir or Path(config.get("index_dir", str(DEFAULT_INDEX_DIR)))
    if not index_dir.is_absolute():
        index_dir = PROJECT_ROOT / index_dir

    logger.info("Index: %s | Chunks: %d words / %d overlap | Model: %s",
                index_dir, chunk_size, overlap, embedding_model)

    files = discover_normalized_files(args.datasets_dir, args.dataset)
    if not files:
        logger.error("No normalized JSONL files found in %s", args.datasets_dir)
        return 1
    logger.info("Found %d normalized file(s)", len(files))

    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir))
    logger.info("Loading embedding model: %s", embedding_model)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    built, skipped = process_files(
        files, client, embedding_fn, chunk_size, overlap, args.force)
    logger.info("Done. Built %d, skipped %d.", built, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
