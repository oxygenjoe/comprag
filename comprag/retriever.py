#!/usr/bin/env python3
"""ChromaDB query interface for the CUMRAG eval harness.

Retrieves top-k chunks from pre-built ChromaDB vector indexes. The retrieval
pipeline is CONSTANT across all eval runs — same embedding model, same index,
same parameters. Only the generator model varies.

Usage (module):
    from comprag.retriever import Retriever
    r = Retriever(index_dir="index", dataset="rgb")
    results = r.retrieve("What is the capital of France?", top_k=5)
    context = r.format_context(results)

Usage (CLI):
    python -m comprag.retriever --query "What is France?" --dataset rgb --top-k 5
    python -m comprag.retriever --dataset rgb --info
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from comprag.utils import get_logger, load_config, make_collection_name, setup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_DATASETS = ("rgb", "nq", "halueval")
DEFAULT_INDEX_DIR = _PROJECT_ROOT / "index"
DEFAULT_TOP_K = 5


def _get_default_embedding_model() -> str:
    """Read embedding model name from eval_config.yaml (single source of truth)."""
    try:
        config = load_config("eval_config")
        return config["retrieval"]["embedding_model"]
    except (FileNotFoundError, KeyError):
        return "all-MiniLM-L6-v2"

logger = get_logger("comprag.retriever")


# ---------------------------------------------------------------------------
# Index validation and collection resolution
# ---------------------------------------------------------------------------


def validate_index(collection, config: dict) -> None:
    """Check that a ChromaDB collection's metadata matches eval_config parameters.

    Ensures the index was built with the same embedding model and chunk size
    as the current configuration. Raises immediately on mismatch to prevent
    silent evaluation against a stale or mismatched index.

    Args:
        collection: A ChromaDB collection object (must have .metadata attribute).
        config: Parsed eval_config.yaml dict (must contain retrieval.embedding_model
                and retrieval.chunk_size).

    Raises:
        ValueError: If embedding_model or chunk_size_words in collection metadata
                    does not match the config values.
    """
    meta = collection.metadata or {}
    expected_model = config["retrieval"]["embedding_model"]
    expected_chunk = config["retrieval"]["chunk_size"]

    if meta.get("embedding_model") != expected_model:
        raise ValueError(
            f"Index embedding model mismatch: {meta.get('embedding_model')} "
            f"vs config {expected_model}. Rebuild index."
        )
    if meta.get("chunk_size_words") != expected_chunk:
        raise ValueError(
            f"Index chunk size mismatch: {meta.get('chunk_size_words')} "
            f"vs config {expected_chunk}. Rebuild index."
        )


def resolve_collection_name(dataset_key: str, config: dict) -> str:
    """Resolve a collection name from eval_config, expanding "auto" entries.

    Looks up ``config["retrieval"]["collections"][dataset_key]``. If the value
    is ``"auto"``, computes a content-addressed name via ``make_collection_name()``
    using the config's embedding_model, chunk_size, and overlap. Otherwise
    returns the explicit collection name from the config.

    Args:
        dataset_key: Key into config["retrieval"]["collections"]
                     (e.g. "rgb_noise_robustness", "nq_wiki").
        config: Parsed eval_config.yaml dict.

    Returns:
        Resolved collection name string.

    Raises:
        KeyError: If dataset_key is not found in config["retrieval"]["collections"].
    """
    collections = config["retrieval"]["collections"]
    value = collections[dataset_key]

    if value == "auto":
        return make_collection_name(
            dataset=dataset_key,
            embedding_model=config["retrieval"]["embedding_model"],
            chunk_size=config["retrieval"]["chunk_size"],
            chunk_overlap=config["retrieval"]["overlap"],
        )
    return value


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """ChromaDB query interface for pre-built vector indexes.

    Loads a persisted ChromaDB collection and provides retrieval methods
    for the eval harness. Uses the same embedding model (all-MiniLM-L6-v2)
    as the indexing pipeline to ensure query/document embedding consistency.

    Attributes:
        index_dir: Path to the ChromaDB persistence directory.
        dataset: Dataset name (rgb, nq, halueval).
        collection_name: ChromaDB collection name (e.g. "comprag_rgb").
        embedding_model: Sentence-transformers model name for query embedding.
    """

    def __init__(
        self,
        index_dir: str = "index",
        dataset: str = "rgb",
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Initialize the retriever with a ChromaDB collection.

        Args:
            index_dir: Path to the ChromaDB persistence directory.
                       Relative paths resolve from the project root.
            dataset: Dataset name. Must be one of: rgb, nq, halueval.
                     Used for fallback collection naming as "comprag_{dataset}"
                     when collection_name is not provided.
            embedding_model: Sentence-transformers model for query embedding.
                             If None, reads from collection metadata first,
                             then falls back to eval_config.yaml.
            collection_name: Explicit ChromaDB collection name (e.g. a
                             content-addressed name from resolve_collection_name).
                             If None, falls back to "comprag_{dataset}".
            config: Parsed eval_config.yaml dict. When provided, validate_index()
                    is called after loading the collection to ensure metadata
                    matches config parameters.

        Raises:
            FileNotFoundError: If the index directory does not exist.
            ValueError: If the dataset is invalid, the collection is not found,
                        the collection is empty, or index metadata mismatches config.
        """
        self.dataset = dataset
        self.collection_name = collection_name if collection_name else f"comprag_{dataset}"

        # Resolve index directory
        index_path = Path(index_dir)
        if not index_path.is_absolute():
            index_path = _PROJECT_ROOT / index_path
        self.index_dir = index_path

        # Validate dataset name
        if dataset not in VALID_DATASETS:
            raise ValueError(
                f"Invalid dataset '{dataset}'. Must be one of: {', '.join(VALID_DATASETS)}"
            )

        # Validate index directory exists
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"Index directory not found: {self.index_dir}\n"
                f"Run 'python scripts/build_index.py --dataset {dataset}' to build the index first."
            )

        # Connect to ChromaDB
        import chromadb

        self._client = chromadb.PersistentClient(path=str(self.index_dir))

        # Validate collection exists
        available = self._list_collection_names()
        if self.collection_name not in available:
            available_str = ", ".join(available) if available else "(none)"
            raise ValueError(
                f"Collection '{self.collection_name}' not found in {self.index_dir}.\n"
                f"Available collections: {available_str}\n"
                f"Run 'python scripts/build_index.py --dataset {dataset}' to build the index."
            )

        # Resolve embedding model: explicit arg > collection metadata > config
        collection_meta = self._client.get_collection(self.collection_name).metadata or {}
        meta_model = collection_meta.get("embedding_model")
        config_model = _get_default_embedding_model()

        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif meta_model:
            self.embedding_model = meta_model
        else:
            self.embedding_model = config_model

        # Warn on mismatch between collection metadata and config
        if meta_model and meta_model != config_model:
            logger.warning(
                "Embedding model mismatch: collection metadata says '%s', "
                "eval_config.yaml says '%s'. Using collection metadata.",
                meta_model,
                config_model,
            )

        # Validate index metadata against config (if config provided)
        if config is not None:
            validate_index(
                self._client.get_collection(self.collection_name), config
            )

        # Load the embedding function (same model as indexing)
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model,
        )

        # Get collection with embedding function
        self._collection = self._client.get_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
        )

        # Validate collection is not empty
        count = self._collection.count()
        if count == 0:
            raise ValueError(
                f"Collection '{self.collection_name}' exists but is empty.\n"
                f"Run 'python scripts/build_index.py --dataset {dataset} --force' to rebuild."
            )

        logger.info(
            "Retriever initialized: collection='%s', chunks=%d, model='%s'",
            self.collection_name,
            count,
            self.embedding_model,
        )

    def _list_collection_names(self) -> list[str]:
        """List all collection names in the ChromaDB instance.

        Returns:
            Sorted list of collection name strings.
        """
        collections = self._client.list_collections()
        names = []
        for c in collections:
            if isinstance(c, str):
                names.append(c)
            elif hasattr(c, "name"):
                names.append(c.name)
        return sorted(names)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Retrieve top-k chunks for a single query.

        Embeds the query using the same model as the index, then performs
        approximate nearest neighbor search in ChromaDB.

        Args:
            query: Query text string. If empty, returns an empty list.
            top_k: Number of chunks to retrieve (default 5).

        Returns:
            List of dicts, ordered by relevance (closest first), each with:
                - text (str): The chunk text.
                - metadata (dict): Chunk metadata with keys: source, chunk_index, dataset.
                - distance (float): L2 distance from query embedding (lower = more relevant).

            Returns an empty list if query is empty or whitespace-only.
        """
        if not query or not query.strip():
            return []

        top_k = max(1, min(top_k, self._collection.count()))

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results, result_index=0)

    def retrieve_batch(
        self, queries: list[str], top_k: int = DEFAULT_TOP_K
    ) -> list[list[dict]]:
        """Batch retrieval for multiple queries.

        More efficient than calling retrieve() in a loop because ChromaDB
        can batch the embedding computation and search.

        Args:
            queries: List of query strings. Empty strings produce empty result lists.
            top_k: Number of chunks to retrieve per query (default 5).

        Returns:
            List of result lists, one per query. Each inner list has the same
            format as retrieve() output.
        """
        if not queries:
            return []

        # Separate non-empty queries from empty ones, preserving order
        non_empty_indices = []
        non_empty_queries = []
        for i, q in enumerate(queries):
            if q and q.strip():
                non_empty_indices.append(i)
                non_empty_queries.append(q)

        # Initialize all results as empty
        all_results: list[list[dict]] = [[] for _ in queries]

        if not non_empty_queries:
            return all_results

        top_k = max(1, min(top_k, self._collection.count()))

        results = self._collection.query(
            query_texts=non_empty_queries,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Map results back to original indices
        for batch_idx, original_idx in enumerate(non_empty_indices):
            all_results[original_idx] = self._format_results(results, result_index=batch_idx)

        return all_results

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string for prompt injection.

        Joins chunk texts with double newlines, suitable for insertion into
        the prompt template's {retrieved_chunks} placeholder.

        Args:
            chunks: List of chunk dicts (as returned by retrieve/retrieve_batch).

        Returns:
            Formatted context string. Empty string if no chunks provided.
        """
        if not chunks:
            return ""

        parts = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if text:
                parts.append(text)

        return "\n\n".join(parts)

    def get_collection_info(self) -> dict:
        """Return collection statistics and metadata.

        Returns:
            Dict with keys:
                - collection_name (str): Name of the collection.
                - count (int): Number of chunks in the collection.
                - dataset (str): Dataset name.
                - embedding_model (str): Embedding model used.
                - metadata (dict): Collection-level metadata from ChromaDB.
        """
        count = self._collection.count()
        metadata = self._collection.metadata or {}

        return {
            "collection_name": self.collection_name,
            "count": count,
            "dataset": self.dataset,
            "embedding_model": self.embedding_model,
            "metadata": metadata,
        }

    @staticmethod
    def _format_results(results: dict, result_index: int = 0) -> list[dict]:
        """Convert ChromaDB query results to our standard output format.

        Args:
            results: Raw ChromaDB query results dict with keys:
                     ids, documents, metadatas, distances.
            result_index: Index into the batch (for batch queries).

        Returns:
            List of formatted chunk dicts with text, metadata, distance.
        """
        documents = results.get("documents", [[]])[result_index] or []
        metadatas = results.get("metadatas", [[]])[result_index] or []
        distances = results.get("distances", [[]])[result_index] or []

        formatted = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Normalize metadata to expected keys
            chunk_meta = {
                "source": meta.get("source", meta.get("doc_id", "unknown")),
                "chunk_index": meta.get("chunk_index", 0),
                "dataset": meta.get("dataset", "unknown"),
            }

            formatted.append({
                "text": doc or "",
                "metadata": chunk_meta,
                "distance": float(dist),
            })

        return formatted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Query the CUMRAG ChromaDB vector index. "
            "Retrieves top-k chunks for a given query from a pre-built index."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  %(prog)s --query "What is the capital of France?" --dataset rgb\n'
            '  %(prog)s --query "quantum computing" --dataset nq --top-k 10\n'
            "  %(prog)s --dataset rgb --info\n"
        ),
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query text to retrieve chunks for.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=VALID_DATASETS,
        default="rgb",
        help="Dataset collection to query (default: rgb).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="index",
        help="Path to ChromaDB persistence directory (default: index/).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model name (default: from collection metadata or eval_config.yaml).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print collection info and exit (no query needed).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    if not args.info and args.query is None:
        parser.error("--query is required unless --info is specified.")

    return args


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 on failure.
    """
    args = parse_args(argv)

    import logging as _logging

    level = _logging.DEBUG if args.verbose else _logging.INFO
    setup_logging(level=level)

    try:
        retriever = Retriever(
            index_dir=args.index_dir,
            dataset=args.dataset,
            embedding_model=args.embedding_model,
        )

        # Info mode: print collection stats and exit
        if args.info:
            info = retriever.get_collection_info()
            print("=" * 60)
            print("Collection Info")
            print("=" * 60)
            print(f"  Collection:      {info['collection_name']}")
            print(f"  Dataset:         {info['dataset']}")
            print(f"  Chunks:          {info['count']}")
            print(f"  Embedding Model: {info['embedding_model']}")
            if info["metadata"]:
                print("  Metadata:")
                for k, v in info["metadata"].items():
                    print(f"    {k}: {v}")
            print("=" * 60)
            return 0

        # Query mode: retrieve and print results
        results = retriever.retrieve(args.query, top_k=args.top_k)

        print("=" * 60)
        print(f"Query: {args.query}")
        print(f"Dataset: {args.dataset} | Top-K: {args.top_k} | Results: {len(results)}")
        print("=" * 60)

        for i, chunk in enumerate(results, start=1):
            meta = chunk["metadata"]
            print(f"\n--- Chunk {i} (distance: {chunk['distance']:.4f}) ---")
            print(f"Source: {meta['source']} | Index: {meta['chunk_index']} | Dataset: {meta['dataset']}")
            print()
            # Truncate long texts for display
            text = chunk["text"]
            if len(text) > 500:
                text = text[:500] + "..."
            print(text)

        print("\n" + "=" * 60)

        # Also print formatted context
        context = retriever.format_context(results)
        if context:
            print("\nFormatted context for prompt injection:")
            print("-" * 40)
            preview = context[:1000] + "..." if len(context) > 1000 else context
            print(preview)

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
