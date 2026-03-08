"""ChromaDB retrieval wrapper for CompRAG.

Connects to a persistent ChromaDB store and returns top-k chunks
as plain strings for a given query and collection.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)


class Retriever:
    """Thin wrapper around ChromaDB for retrieval.

    Index building is handled by scripts/build_index.py — this class
    only reads from an existing persistent store.
    """

    def __init__(self, index_dir: str, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        """Initialize retriever with a persistent ChromaDB directory.

        Args:
            index_dir: Path to the ChromaDB persistent storage directory.
            embedding_model: Sentence-transformer model name for query embedding.
        """
        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory does not exist: {index_dir}")

        logger.info("Loading embedding model: %s", embedding_model)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )

        logger.info("Connecting to ChromaDB at: %s", index_dir)
        self._client = chromadb.PersistentClient(path=str(self.index_dir))
        self._collection_names = [c.name for c in self._client.list_collections()]
        logger.info("Retriever ready, %d collections", len(self._collection_names))

    def query(self, text: str, collection: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k chunks matching a query from a collection.

        Args:
            text: The query string to search for.
            collection: Name of the ChromaDB collection to search.
            top_k: Number of chunks to return. Default 5 (locked in eval.yaml).

        Returns:
            List of chunk text strings, ordered by relevance (most relevant first).
        """
        resolved = self._resolve_collection(collection)
        col = self._client.get_collection(
            name=resolved,
            embedding_function=self._embedding_fn,
        )

        results = col.query(
            query_texts=[text],
            n_results=top_k,
        )

        documents = results.get("documents")
        if not documents or not documents[0]:
            logger.warning("No results for query in collection '%s': %s", collection, text[:80])
            return []

        chunks = documents[0]
        logger.info(
            "Retrieved %d chunks from '%s' for query: %s",
            len(chunks),
            collection,
            text[:80],
        )
        return chunks

    def _resolve_collection(self, name: str) -> str:
        """Resolve a logical collection name to the actual ChromaDB name.

        Handles the cumrag_<name>_300w_<hash> naming convention from
        build_index.py. Tries exact match, then stripped-prefix match,
        then substring match on the dataset portion.
        """
        if name in self._collection_names:
            return name
        # Strip cumrag_ prefix and _300w_<hash> suffix, compare to logical name
        for actual in self._collection_names:
            core = actual.replace("cumrag_", "", 1).rsplit("_300w_", 1)[0]
            if core == name:
                logger.info("Resolved collection '%s' -> '%s'", name, actual)
                return actual
        # Fallback: match by dataset prefix (e.g. "nq" matches "cumrag_nq_wiki_...")
        matches = [a for a in self._collection_names if f"_{name}_" in a or a.startswith(f"cumrag_{name}_")]
        if len(matches) == 1:
            logger.info("Resolved collection '%s' -> '%s' (prefix match)", name, matches[0])
            return matches[0]
        raise ValueError(
            f"Collection '{name}' not found. Available: {self._collection_names}"
        )

    def list_collections(self) -> list[str]:
        """Return names of all available collections."""
        return list(self._collection_names)
