from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMADB = True
except Exception:  # pragma: no cover - optional dependency fallback
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    _HAS_CHROMADB = False

from src.config import VECTORSTORE_DIR

logger = logging.getLogger(__name__)

_ID_LOOKUP_BATCH_SIZE = 500


class EmbeddingDimensionMismatchError(ValueError):
    """Raised when new embeddings are incompatible with an existing collection."""


def _embedding_dimension(embedding: List[float]) -> int:
    return len(embedding)


def _is_dimension_mismatch_error(exc: Exception) -> bool:
    return "dimension" in str(exc).lower()


class ChromaVectorStore:
    """Vector database manager using ChromaDB."""

    def __init__(self, collection_name: str = "knowledge_base", persist_directory: str | Path = VECTORSTORE_DIR):
        if not _HAS_CHROMADB:
            raise ImportError(
                "chromadb is required for the vector store. Install with `pip install chromadb`"
            )
        self.persist_directory = str(Path(persist_directory).resolve())
        if hasattr(chromadb, "PersistentClient"):
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False) if Settings else None,
            )
        else:
            self.client = chromadb.Client(Settings(persist_directory=self.persist_directory, anonymized_telemetry=False))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        # expose `index` for compatibility with earlier vector-store implementations
        # return the collection object when there are any items, else None
        self._index_cached: Optional[Any] = None

    def get_collection_embedding_dimension(self) -> Optional[int]:
        """Return the embedding dimension of one stored vector, if any."""
        if self.get_collection_count() == 0:
            return None
        try:
            sample = self.collection.get(limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                return None
            first = embeddings[0]
            if first is None:
                return None
            return _embedding_dimension(list(first))
        except Exception as exc:
            logger.warning("Could not read collection embedding dimension: %s", exc)
            return None

    def _validate_new_embeddings(self, embeddings: List[List[float]]) -> None:
        """Ensure new embeddings are compatible with the existing collection."""
        if not embeddings:
            return

        new_dims = {_embedding_dimension(embedding) for embedding in embeddings}
        if len(new_dims) != 1:
            raise ValueError(
                f"Inconsistent embedding dimensions within batch: {sorted(new_dims)}"
            )
        new_dim = next(iter(new_dims))

        existing_dim = self.get_collection_embedding_dimension()
        if existing_dim is not None and existing_dim != new_dim:
            raise EmbeddingDimensionMismatchError(
                "Embedding dimension mismatch: the existing Chroma collection "
                f"uses dimension {existing_dim}, but the new embeddings have "
                f"dimension {new_dim}. The collection was NOT modified. "
                "Your embedding configuration (EMBEDDING_MODEL, LLM_BASE_URL, "
                "and DEV_EMBEDDINGS) must match the configuration used to build "
                "the existing index. Do not mix synthetic/dev embeddings with "
                "production API embeddings in the same collection."
            )

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]

        self._validate_new_embeddings(embeddings)

        try:
            if hasattr(self.collection, "upsert"):
                self.collection.upsert(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
            else:
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
        except Exception as exc:
            if _is_dimension_mismatch_error(exc):
                existing_dim = self.get_collection_embedding_dimension()
                new_dim = _embedding_dimension(embeddings[0]) if embeddings else None
                raise EmbeddingDimensionMismatchError(
                    "Chroma rejected embeddings due to a dimension mismatch. "
                    f"Existing collection dimension: {existing_dim}. "
                    f"New embedding dimension: {new_dim}. "
                    "The collection was NOT deleted and was NOT modified. "
                    "Align EMBEDDING_MODEL, LLM_BASE_URL, and DEV_EMBEDDINGS "
                    "with the configuration used for the existing vectors."
                ) from exc
            raise

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Query the collection and return raw Chroma results."""
        query_dim = _embedding_dimension(query_embedding)
        existing_dim = self.get_collection_embedding_dimension()
        if existing_dim is not None and query_dim != existing_dim:
            raise EmbeddingDimensionMismatchError(
                f"Query embedding dimension {query_dim} does not match "
                f"collection dimension {existing_dim}."
            )
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results

    def delete_collection(self) -> None:
        """Explicit collection deletion — never called automatically by this class."""
        self.client.delete_collection(name=self.collection_name)

    def get_collection_count(self) -> int:
        return self.collection.count()

    def collection_exists(self) -> bool:
        """Return True if the named collection is registered in Chroma."""
        try:
            self.client.get_collection(name=self.collection_name)
            return True
        except Exception:
            return False

    def get_existing_ids(self, ids: List[str]) -> Set[str]:
        """Return the subset of *ids* that already exist in the collection."""
        if not ids:
            return set()

        existing: Set[str] = set()
        for start in range(0, len(ids), _ID_LOOKUP_BATCH_SIZE):
            batch = ids[start : start + _ID_LOOKUP_BATCH_SIZE]
            try:
                result = self.collection.get(ids=batch, include=[])
            except Exception as exc:
                logger.warning("get_existing_ids batch lookup failed: %s", exc)
                continue
            if result and result.get("ids"):
                existing.update(result["ids"])
        return existing

    def get_diagnostics(
        self,
        *,
        configured_model: Optional[str] = None,
        configured_base_url: Optional[str] = None,
        dev_embeddings: bool = False,
        current_embedding_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return safe, non-secret vector-store diagnostics."""
        provider_host = "unknown"
        if configured_base_url:
            provider_host = urlparse(configured_base_url).netloc or configured_base_url

        existing_dim = self.get_collection_embedding_dimension()
        count = self.get_collection_count()
        dimensions_compatible: Optional[bool] = None
        if existing_dim is not None and current_embedding_dim is not None:
            dimensions_compatible = existing_dim == current_embedding_dim

        return {
            "collection_name": self.collection_name,
            "collection_exists": self.collection_exists(),
            "collection_count": count,
            "existing_embedding_dimension": existing_dim,
            "current_embedding_dimension": current_embedding_dim,
            "dimensions_compatible": dimensions_compatible,
            "configured_embedding_provider": provider_host,
            "configured_embedding_model": configured_model,
            "dev_embeddings": dev_embeddings,
            "persist_directory": self.persist_directory,
        }

    @property
    def index(self) -> Optional[Any]:
        try:
            cnt = self.get_collection_count()
            return self.collection if cnt and cnt > 0 else None
        except Exception:
            return None
