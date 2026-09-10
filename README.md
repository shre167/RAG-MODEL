from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rank_bm25 import BM25Plus as BM25Model
except ImportError:
    from rank_bm25 import BM25Okapi as BM25Model

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, VECTORSTORE_DIR
from src.document_loader import chunk_documents, load_documents

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Lightweight local BM25 retriever built from chunked knowledge base.

    Matches the same chunk boundaries as ChromaVectorStore (CHUNK_SIZE=800,
    CHUNK_OVERLAP=200) so RRF fusion receives candidates from identical chunks.

    Supports:
    - Disk persistence via save_index / load_index (pickle).
    - Incremental indexing via add_chunks() so newly-uploaded files are
      queryable immediately without re-reading the full corpus.
    - Full section metadata passed through to rag_pipeline.py so
      section_heading, section_path, section_level, etc. are available.
    """

    _INDEX_FILENAME = "bm25_index.pkl"

    def __init__(
        self,
        kb_path: str | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        vectorstore_dir: str | Path | None = None,
    ):
        self.kb_path = kb_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore_dir = Path(vectorstore_dir or VECTORSTORE_DIR)

        self._chunks: List[Dict[str, Any]] = []
        self._tokenized: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    # ------------------------------------------------------------------
    # Index path
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.vectorstore_dir / self._INDEX_FILENAME

    # ------------------------------------------------------------------
    # Build / Rebuild
    # ------------------------------------------------------------------

    def build_index(
        self,
        kb_path: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Load all documents from kb_path, chunk them, and build BM25 index."""
        path = kb_path or self.kb_path
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap

        docs = load_documents(path)
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self._chunks = []
        self._tokenized = []

        for c in chunks:
            tokens = self._tokenize(c.get("text") or "")
            self._tokenized.append(tokens)
            self._chunks.append(self._extract_chunk_meta(c))

        if self._tokenized:
            self._bm25 = BM25Model(self._tokenized)

        logger.info(
            "BM25: built index over %d chunks from %d documents.",
            len(self._chunks),
            len(docs),
        )

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> None:
        """Append new pre-chunked documents to the existing BM25 index.

        ``new_chunks`` should be the same dict format produced by
        ``chunk_documents()`` — each item must have at least a ``"text"`` key.
        """
        if not new_chunks:
            return

        for c in new_chunks:
            tokens = self._tokenize(c.get("text") or "")
            self._tokenized.append(tokens)
            self._chunks.append(self._extract_chunk_meta(c))

        # Rebuild BM25 over all (existing + new) tokens.
        if self._tokenized:
            self._bm25 = BM25Model(self._tokenized)

        logger.info(
            "BM25: added %d chunks; index now contains %d chunks.",
            len(new_chunks),
            len(self._chunks),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path | None = None) -> None:
        """Persist the BM25 corpus and chunk metadata to disk."""
        target = Path(path) if path else self._index_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": self._chunks,
            "tokenized": self._tokenized,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        with open(target, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25: index saved to %s.", target)

    def load_index(self, path: str | Path | None = None) -> bool:
        """Restore index from disk.  Returns True on success, False otherwise."""
        target = Path(path) if path else self._index_path()
        if not target.exists():
            return False
        try:
            with open(target, "rb") as fh:
                payload = pickle.load(fh)
            self._chunks = payload.get("chunks", [])
            self._tokenized = payload.get("tokenized", [])
            self.chunk_size = payload.get("chunk_size", self.chunk_size)
            self.chunk_overlap = payload.get("chunk_overlap", self.chunk_overlap)
            if self._tokenized:
                self._bm25 = BM25Model(self._tokenized)
            logger.info(
                "BM25: loaded index from %s (%d chunks).",
                target,
                len(self._chunks),
            )
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("BM25: failed to load index from %s: %s", target, exc)
            return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return up to top_k chunks sorted by BM25 score (descending)."""
        if not self._bm25 or not self._chunks:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in ranked_idx:
            score = float(scores[idx])
            if score <= 0:
                # Token overlap fallback for small corpora
                overlap = set(tokens) & set(self._tokenized[idx])
                if not overlap:
                    continue
                score = float(len(overlap))
            chunk = self._chunks[idx]
            results.append({
                **chunk,
                "bm25_score": score,
            })
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return (text or "").lower().split()

    @staticmethod
    def _extract_chunk_meta(c: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all metadata fields that rag_pipeline.py expects."""
        return {
            "filename": c.get("filename", ""),
            "chunk_id": c.get("chunk_id", ""),
            "text": c.get("text", ""),
            "source_path": c.get("source_path", ""),
            "section_heading": c.get("section_heading", ""),
            "section_path": c.get("section_path", ""),
            "section_level": c.get("section_level", 0),
            "chunk_index": c.get("chunk_index", 0),
            "total_section_chunks": c.get("total_section_chunks", 0),
            "category": c.get("category", ""),
            "has_heading": c.get("has_heading", False),
        }
