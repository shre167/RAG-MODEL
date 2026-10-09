"""
Unified Vector Store - Automatically uses PostgreSQL if available, falls back to Chroma.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import USE_POSTGRES, VECTORSTORE_DIR

logger = logging.getLogger(__name__)

class UnifiedVectorStore:
    """Auto-selects PostgreSQL or Chroma based on configuration."""
    
    def __init__(self, collection_name: str = "knowledge_base", persist_directory: str | Path = VECTORSTORE_DIR):
        self.collection_name = collection_name
        self.persist_directory = str(Path(persist_directory).resolve())
        
        # Try PostgreSQL first if configured
        if USE_POSTGRES:
            try:
                from src.postgres_vector_store_simple import PostgresVectorStore
                self._backend = PostgresVectorStore(collection_name, persist_directory)
                self._backend_name = "postgres"
                logger.info("Using PostgreSQL vector store backend")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL backend: {e}. Falling back to Chroma.")
        
        # Fall back to Chroma
        try:
            from src.vector_store import ChromaVectorStore
            self._backend = ChromaVectorStore(collection_name, persist_directory)
            self._backend_name = "chroma"
            logger.info("Using Chroma vector store backend")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma backend: {e}")
            raise
    
    def get_collection_embedding_dimension(self) -> Optional[int]:
        return self._backend.get_collection_embedding_dimension()
    
    def _validate_new_embeddings(self, embeddings: List[List[float]]) -> None:
        self._backend._validate_new_embeddings(embeddings)
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        self._backend.add_documents(texts, embeddings, metadatas, ids)
    
    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        return self._backend.query(query_embedding, n_results)
    
    def delete_collection(self) -> None:
        self._backend.delete_collection()
    
    def clear_collection(self) -> None:
        self._backend.clear_collection()
    
    def get_collection_count(self) -> int:
        return self._backend.get_collection_count()
    
    def collection_exists(self) -> bool:
        return self._backend.collection_exists()
    
    def get_existing_ids(self, ids: List[str]) -> Set[str]:
        return self._backend.get_existing_ids(ids)
    
    def get_diagnostics(
        self,
        *,
        configured_model: Optional[str] = None,
        configured_base_url: Optional[str] = None,
        current_embedding_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        diag = self._backend.get_diagnostics(
            configured_model=configured_model,
            configured_base_url=configured_base_url,
            current_embedding_dim=current_embedding_dim
        )
        diag["backend"] = self._backend_name
        return diag
    
    @property
    def index(self):
        return self._backend.index if hasattr(self._backend, 'index') else None
    
    @property
    def collection(self):
        return self._backend.collection if hasattr(self._backend, 'collection') else self._backend

