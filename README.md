from __future__ import annotations

import logging
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    GE_API_KEY,
    KNOWLEDGE_BASE_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    MIN_RERANK_SCORE,
    RERANK_TOP_K,
    SCORE_THRESHOLD,
    TOP_K,
    VECTORSTORE_DIR,
    BM25_TOP_K,
    HYBRID_TOP_K,
    VECTOR_TOP_K,
)

from src.document_loader import (
    chunk_documents,
    list_txt_files,
    load_documents,
)
from src.embeddings import EmbeddingService
from src.bm25_retriever import BM25Retriever
from src.query_normalizer import canonicalize_query
from src.utils import ensure_directory
from src.vector_store import ChromaVectorStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================================
# GENERAL SETTINGS
# ======================================================================

ABSTAIN_MESSAGE = (
    "I couldn't find enough relevant information in the knowledge base "
    "to answer that reliably."
)


# ======================================================================
# RETRIEVAL SETTINGS
# ======================================================================

# Reciprocal Rank Fusion constant.
# RRF is based on rank rather than raw score, so dense and BM25 scores
# do not need to be on the same numerical scale.
RRF_K = 60


# Retrieve substantially more candidates than are finally sent to the LLM.
#
# This prevents the first retriever ranking from becoming the final answer
# ranking, especially as the knowledge base grows.
MIN_RERANK_CANDIDATES = 30
MAX_RERANK_CANDIDATES = 50

RERANK_CANDIDATE_LIMIT = min(
    max(
        HYBRID_TOP_K,
        VECTOR_TOP_K,
        BM25_TOP_K,
        MIN_RERANK_CANDIDATES,
    ),
    MAX_RERANK_CANDIDATES,
)


# Number of chunks that can ultimately reach the LLM.
#
# Keeping this bounded prevents context bloat as the KB grows.
MAX_CONTEXT_CHUNKS = min(
    max(RERANK_TOP_K, 5),
    8,
)


# Prevent one source file from dominating the final context.
MAX_CHUNKS_PER_SOURCE = 3


# Maximum characters sent to the LLM as retrieved context.
MAX_CONTEXT_CHARS = 18000


# ======================================================================
# RERANKING / EVIDENCE SETTINGS
# ======================================================================

# IMPORTANT:
#
# CrossEncoder scores are raw model scores.
# They are NOT probabilities.
#
# MIN_RERANK_SCORE should therefore be treated as a configurable
# absolute sanity floor and calibrated against an evaluation dataset.
RERANK_ABSTAIN_FLOOR = MIN_RERANK_SCORE

# CrossEncoder reranking is intentionally disabled.
# Retrieval uses Dense + BM25 + RRF only.
ENABLE_RERANKER = False

# RRF is a ranking signal, not a probability.
# With RRF_K=60, a rank-0 result in both retrievers is ~0.03279.
STRONG_RRF_THRESHOLD = 0.025
MODERATE_RRF_THRESHOLD = RRF_ABSTAIN_FLOOR


# Relative separation from the runner-up.
#
# These are intentionally separate from the support-window threshold.
MIN_RERANK_GAP = 0.25
STRONG_RERANK_GAP = 0.75


# A candidate can be considered supporting context when its score is
# sufficiently close to the best candidate.
#
# This is NOT an answer-confidence threshold.
SUPPORT_SCORE_WINDOW = 1.0


# Percentile is used as a secondary distribution signal.
#
# Unlike the broken implementation that compared every score to max(scores),
# this calculation measures where the top score sits relative to the
# actual score distribution.
MIN_TOP_PERCENTILE = 0.70
STRONG_TOP_PERCENTILE = 0.85


# Minimum number of candidates used when calculating a distribution signal.
# With very small pools, percentile statistics are unstable.
MIN_PERCENTILE_POOL = 5


# RRF is a rank-based score, not a calibrated relevance probability.
#
# This is only a conservative fallback threshold when CrossEncoder
# reranking is unavailable.
RRF_ABSTAIN_FLOOR = 0.015


# ======================================================================
# GENERIC RAG SYSTEM PROMPT
# ======================================================================

SYSTEM_PROMPT = """
You are a Knowledge Base Assistant.

Your job is to answer the user's question using ONLY the
retrieved information supplied in the conversation.

Rules:

1. Answer directly and clearly.

2. Use only information supported by the retrieved context.

3. Do not use outside knowledge to fill missing information.

4. Do not invent facts, numbers, names, dates, procedures,
   explanations, or examples that are not supported by the context.

5. You may combine multiple retrieved chunks only when they
   directly support the same answer.

6. If the retrieved information is insufficient, say so clearly.

7. If the user asks for a complete list and the retrieved context
   contains only part of the list, explicitly say that the available
   information is partial. Do not invent the missing items.

8. Prefer concise answers unless the user asks for a detailed explanation.

9. For processes or procedures, use numbered steps.

10. Preserve important technical terminology from the retrieved context.

11. Do not mention retrieval scores, embeddings, BM25, reranking,
    vector databases, or internal pipeline details unless the user
    explicitly asks about them.

12. Sources must contain only filenames that are explicitly present
    in the retrieved information.

13. Do not create or guess source filenames.

Response format:

Answer:
<direct answer>

Details:
<supporting explanation if useful>

Sources:
<source filenames>
"""


# ======================================================================
# DATA CLASSES
# ======================================================================

@dataclass
class Candidate:
    """
    Represents one retrieved chunk throughout the retrieval pipeline.

    Flow:

        Dense Retrieval
              +
        BM25 Retrieval
              ↓
          RRF Fusion
              ↓
        Candidate Pool
              ↓
        Evidence Evaluation
              ↓
        Context Selection
    """

    key: str
    filename: str
    chunk_id: Any
    text: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    dense_rank: int | None = None
    dense_distance: float | None = None

    bm25_rank: int | None = None
    bm25_score: float | None = None

    rrf_score: float = 0.0

    rerank_score: float | None = None

    @property
    def in_both_retrievers(self) -> bool:
        return (
            self.dense_rank is not None
            and self.bm25_rank is not None
        )


@dataclass
class Evidence:
    """
    Represents the strength of the retrieved evidence.

    The evidence gate combines:

    - absolute RRF floor
    - RRF rank strength
    - retriever agreement
    - source diversity
    """

    level: str
    should_answer: bool

    top_score: float
    score_source: str

    supporting_chunks: int

    top_gap: float | None

    agreement: bool

    top_percentile: float | None

    retrieval_mode: str

    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "should_answer": self.should_answer,
            "top_score": self.top_score,
            "score_source": self.score_source,
            "supporting_chunks": self.supporting_chunks,
            "top_gap": self.top_gap,
            "retrieval_agreement": self.agreement,
            "top_percentile": self.top_percentile,
            "retrieval_mode": self.retrieval_mode,
            "reason": self.reason,
        }


# ======================================================================
# MAIN PIPELINE
# ======================================================================

class RAGPipeline:
    """
    Domain-independent RAG pipeline.

    Architecture:

        User Query
             ↓
        Query Preparation
             ↓
        ┌──────────────────────┐
        │ Dense Retrieval      │
        │ BM25 Retrieval       │
        └──────────┬───────────┘
                   ↓
              RRF Fusion
                   ↓
            Wider Candidate Pool
                   ↓
            CrossEncoder
             Reranking
                   ↓
          Evidence Evaluation
                   ↓
       Deduplication + Diversity
                   ↓
          Context Selection
                   ↓
                  LLM
                   ↓
              Final Answer
    """

    def __init__(
        self,
        knowledge_base_path: str | Path = KNOWLEDGE_BASE_DIR,
        vectorstore_path: str | Path = VECTORSTORE_DIR,
        model_name: str | None = EMBEDDING_MODEL,
        top_k: int = TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):

        self.knowledge_base_path = Path(
            knowledge_base_path
        )

        self.vectorstore_path = ensure_directory(
            vectorstore_path
        )

        self.top_k = top_k

        # Kept for backwards compatibility with the existing pipeline.
        #
        # Dense distances, BM25 scores, RRF scores and CrossEncoder
        # scores are different score spaces, so one universal
        # SCORE_THRESHOLD should not be applied blindly to all of them.
        self.score_threshold = score_threshold

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # ==============================================================

        # EMBEDDING SERVICE

        # ==============================================================

        if os.getenv(
            "DEV_EMBEDDINGS",
            "false",
        ).lower() in (
            "1",
            "true",
            "yes",
        ):

            from src.embeddings import (
                SyntheticEmbeddingService,
            )

            self.embedding_service = (
                SyntheticEmbeddingService()
            )

        else:

            self.embedding_service = EmbeddingService(
                api_key=GE_API_KEY or LLM_API_KEY,
                model=model_name,
                base_url=LLM_BASE_URL,
            )

        # ==============================================================

        # VECTOR STORE

        # ==============================================================

        self.vector_store = ChromaVectorStore(
            collection_name="knowledge_base",
            persist_directory=self.vectorstore_path,
        )

        try:

            self.index_loaded = (
                self.vector_store
                .get_collection_count()
                > 0
            )

        except Exception:

            self.index_loaded = False

        # ==============================================================

        # BM25

        # ==============================================================

        self.bm25 = None

        try:

            self.bm25 = BM25Retriever(
                kb_path=str(
                    self.knowledge_base_path
                ),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            if self.knowledge_base_path.exists():

                self.bm25.build_index(
                    str(
                        self.knowledge_base_path
                    )
                )

            logger.info(
                "BM25 retriever initialized."
            )

        except Exception as exc:

            logger.warning(
                "BM25 initialization failed: %s",
                exc,
            )

        # ==============================================================
        # CROSS-ENCODER RERANKER
        # ==============================================================
        # Disabled intentionally. Do not instantiate ReRanker here; doing
        # so can trigger a local Hugging Face model download.
        self.reranker = None
        logger.info(
            "CrossEncoder reranker disabled; using Dense + BM25 + RRF only."
        )

    # ==================================================================
    # HELPERS
    # ==================================================================

    @staticmethod
    def _candidate_key(
        filename: str,
        chunk_id: Any,
    ) -> str:

        return (
            f"{filename}"
            f"::chunk-{chunk_id}"
        )

    @staticmethod
    def _safe_float(
        value: Any,
        default: float | None = None,
    ) -> float | None:

        try:

            if value is None:

                return default

            number = float(value)

            if not math.isfinite(number):

                return default

            return number

        except (
            TypeError,
            ValueError,
        ):

            return default

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            (text or "").strip().lower(),
        )

    def _retrieval_mode(
        self,
        candidates: list[Candidate],
    ) -> str:

        has_dense = any(
            candidate.dense_rank is not None
            for candidate in candidates
        )

        has_bm25 = any(
            candidate.bm25_rank is not None
            for candidate in candidates
        )

        if has_dense and has_bm25:

            return "hybrid"

        if has_dense:

            return "dense_only"

        if has_bm25:

            return "bm25_only"

        return "unknown"

    @staticmethod
    def _is_multi_intent_query(question: str) -> bool:
        """Detect simple multi-part questions that need broader source coverage.

        This is intentionally conservative. It does not split or rewrite the
        user's query; it only tells final context selection to preserve
        evidence from more than one source when the wording clearly contains
        multiple requests.
        """
        text = (question or "").strip().lower()
        if not text:
            return False

        if "?" in text and text.count("?") > 1:
            return True

        return bool(
            re.search(
                r"\b(?:and|also|as well as|along with|plus)\b",
                text,
            )
        )

    # ==================================================================
    # QUERY PREPARATION
    # ==================================================================

    @staticmethod
    def _prepare_search_query(
        question: str,
    ) -> str:

        original = question.strip()

        if not original:

            return original

        try:

            normalized = canonicalize_query(
                original
            )

            expanded = (
                normalized.get(
                    "expanded_query",
                    "",
                )
                or ""
            ).strip()

            # Preserve the original user query.
            #
            # We do NOT replace it with the normalized form because
            # normalization can accidentally remove useful terminology.
            if (
                expanded
                and expanded.lower()
                != original.lower()
            ):

                return (
                    f"{original} {expanded}"
                ).strip()

        except Exception as exc:

            logger.debug(
                "Query normalization failed: %s",
                exc,
            )

        return original

    # ==================================================================
    # STATUS
    # ==================================================================

    def status(self) -> dict[str, Any]:

        files = list_txt_files(
            self.knowledge_base_path
        )

        try:

            collection_count = (
                self.vector_store
                .get_collection_count()
            )

        except Exception:

            collection_count = 0

        return {
            "knowledge_base_exists":
                self.knowledge_base_path.exists(),

            "txt_files_found":
                len(files),

            "index_available":
                collection_count > 0,

            "collection_count":
                collection_count,

            "vectorstore_path":
                str(self.vectorstore_path),

            "embedding_model":
                getattr(
                    self.embedding_service,
                    "model",
                    None,
                ),

            "reranker_available":
                (
                    self.reranker is not None
                    and getattr(
                        self.reranker,
                        "available",
                        False,
                    )
                ),

            "bm25_available":
                self.bm25 is not None,

            "retrieval_settings": {
                "vector_top_k":
                    VECTOR_TOP_K,

                "bm25_top_k":
                    BM25_TOP_K,

                "hybrid_top_k":
                    HYBRID_TOP_K,

                "rerank_candidate_limit":
                    RERANK_CANDIDATE_LIMIT,

                "rerank_top_k":
                    RERANK_TOP_K,

                "max_context_chunks":
                    MAX_CONTEXT_CHUNKS,

                "max_chunks_per_source":
                    MAX_CHUNKS_PER_SOURCE,

                "max_context_chars":
                    MAX_CONTEXT_CHARS,
            },
        }

    # ==================================================================
    # INGESTION
    # ==================================================================

    def ingest_documents(self) -> dict[str, Any]:

        if not self.knowledge_base_path.exists():

            self.knowledge_base_path.mkdir(
                parents=True,
                exist_ok=True,
            )

        docs = load_documents(
            self.knowledge_base_path
        )

        if not docs:

            raise ValueError(
                "No .txt files were found in the "
                "knowledge base."
            )

        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Remove empty chunks before embedding.
        chunks = [
            chunk
            for chunk in chunks
            if (
                chunk.get(
                    "text",
                    "",
                )
                or ""
            ).strip()
        ]

        if not chunks:

            raise ValueError(
                "No usable content was found in the "
                "knowledge base after chunking."
            )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            self._build_chunk_metadata(
                chunk
            )
            for chunk in chunks
        ]

        # --------------------------------------------------------------
        # CREATE STABLE IDS
        # --------------------------------------------------------------

        ids = [
            self._candidate_key(
                meta["filename"],
                meta["chunk_id"],
            )
            for meta in metadatas
        ]

        if len(set(ids)) != len(ids):

            raise ValueError(
                "Duplicate chunk IDs were generated "
                "during ingestion."
            )

        if len(ids) != len(texts):

            raise ValueError(
                "ID count mismatch: "
                f"expected {len(texts)}, "
                f"got {len(ids)}"
            )

        count_before = (
            self.vector_store
            .get_collection_count()
        )

        existing_ids = self.vector_store.get_existing_ids(ids)
        missing_indices = [
            index
            for index, chunk_id in enumerate(ids)
            if chunk_id not in existing_ids
        ]
        skipped_count = len(ids) - len(missing_indices)
        batch_size = max(1, EMBEDDING_BATCH_SIZE)
        embedded_count = 0

        logger.info(
            "Ingest: %d documents -> %d chunks. "
            "Existing collection count: %d. "
            "Already indexed: %d. To embed: %d.",

            len(docs),
            len(chunks),
            count_before,
            skipped_count,
            len(missing_indices),
        )

        # --------------------------------------------------------------
        # EMBED + PERSIST MISSING CHUNKS IN BATCHES
        # --------------------------------------------------------------

        for batch_start in range(0, len(missing_indices), batch_size):
            batch_indices = missing_indices[
                batch_start : batch_start + batch_size
            ]
            batch_texts = [texts[index] for index in batch_indices]
            batch_metadatas = [metadatas[index] for index in batch_indices]
            batch_ids = [ids[index] for index in batch_indices]

            try:
                embeddings = (
                    self.embedding_service
                    .create_embeddings_batch(
                        batch_texts
                    )
                )
            except Exception as exc:
                raise ValueError(
                    "Embedding generation failed during ingestion "
                    f"after persisting {embedded_count}/{len(missing_indices)} "
                    f"missing chunks: {exc}"
                ) from exc

            if (
                not embeddings
                or len(embeddings) != len(batch_texts)
            ):
                raise ValueError(
                    "Embedding count mismatch: "
                    f"expected {len(batch_texts)}, "
                    f"got "
                    f"{len(embeddings) if embeddings else 0}"
                )

            self.vector_store.add_documents(
                texts=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
            embedded_count += len(batch_ids)
            logger.info(
                "Persisted embedding batch: %d/%d missing chunks stored.",
                embedded_count,
                len(missing_indices),
            )

        self.index_loaded = True

        # --------------------------------------------------------------
        # RESYNC BM25
        # --------------------------------------------------------------

        if self.bm25 is not None:

            try:

                self.bm25.build_index(
                    str(
                        self.knowledge_base_path
                    )
                )

            except Exception as exc:

                logger.warning(
                    "BM25 resync failed: %s",
                    exc,
                )

        count_after = (
            self.vector_store
            .get_collection_count()
        )

        if count_after == 0:

            raise RuntimeError(
                "Vector store is empty after ingestion."
            )

        return {
            "chunks_indexed": embedded_count,
            "chunks_skipped": skipped_count,
            "chunks_total": len(ids),
            "files_processed": len(docs),
            "chunks_created": len(chunks),
            "collection_count": count_after,
        }

    # ==================================================================
    # INCREMENTAL SINGLE-FILE INGESTION
    # ==================================================================

    def ingest_file(
        self,
        file_path: "str | Path",
        content: "str | None" = None,
    ) -> "dict[str, Any]":
        """Ingest a single file incrementally.

        If ``content`` is provided the file does not need to exist on disk
        yet (useful for Streamlit uploads that arrive as in-memory bytes).

        The file is saved to ``knowledge_base/`` when content is provided
        and the file does not already exist there.  The ChromaDB vector
        store and BM25 index are updated immediately so the new document
        is queryable without a full rebuild.

        Returns a dict with:
            filename, chunks_added, collection_count, skipped (bool)
        """
        import hashlib

        file_path = Path(file_path)

        # ------------------------------------------------------------------
        # Resolve / save content
        # ------------------------------------------------------------------

        if content is not None:
            target = self.knowledge_base_path / file_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            file_path = target
        else:
            if not file_path.is_absolute():
                file_path = self.knowledge_base_path / file_path
            if not file_path.exists():
                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )
            content = file_path.read_text(encoding="utf-8", errors="ignore")

        filename = file_path.name
        file_hash = hashlib.sha256(content.encode()).hexdigest()

        # ------------------------------------------------------------------
        # Manifest check – skip unchanged files
        # ------------------------------------------------------------------

        manifest = self._load_manifest()

        existing = manifest.get(filename, {})
        if existing.get("hash") == file_hash:
            logger.info(
                "ingest_file: %s unchanged (hash match), skipping.",
                filename,
            )
            return {
                "filename": filename,
                "chunks_added": 0,
                "collection_count": self.vector_store.get_collection_count(),
                "skipped": True,
            }

        # ------------------------------------------------------------------
        # Chunk
        # ------------------------------------------------------------------

        from src.document_loader import clean_text as _clean_text

        doc = {
            "filename": filename,
            "source_path": str(file_path),
            "content": _clean_text(content),
        }

        new_chunks = chunk_documents(
            [doc],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        new_chunks = [
            c for c in new_chunks
            if (c.get("text") or "").strip()
        ]

        if not new_chunks:
            raise ValueError(
                f"No usable content found in {filename} after chunking."
            )

        texts = [c["text"] for c in new_chunks]
        metadatas = [self._build_chunk_metadata(c) for c in new_chunks]
        ids = [
            self._candidate_key(m["filename"], m["chunk_id"])
            for m in metadatas
        ]

        # ------------------------------------------------------------------
        # Existing chunks are deliberately NOT deleted yet.
        #
        # The new embeddings must be generated and validated successfully
        # before any old data is removed. This prevents an API failure, quota
        # error, or dimension mismatch from destroying the previous index.
        # ------------------------------------------------------------------

        old_ids = existing.get("chunk_ids", [])

        # ------------------------------------------------------------------
        # Embed
        # ------------------------------------------------------------------

        try:
            embeddings = self.embedding_service.create_embeddings_batch(texts)
        except Exception as exc:
            raise ValueError(
                f"Embedding failed for {filename}: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Store in ChromaDB
        # ------------------------------------------------------------------

        if not embeddings or len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding count mismatch for {filename}: "
                f"expected {len(texts)}, got "
                f"{len(embeddings) if embeddings else 0}"
            )

        # add_documents() validates embedding dimensions BEFORE mutating the
        # collection. Upsert first so unchanged chunk IDs are safely replaced.
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        self.index_loaded = True

        # Remove only chunks that belonged to the previous version and are no
        # longer present. This is done AFTER the new version is safely stored.
        new_id_set = set(ids)
        stale_old_ids = [
            old_id
            for old_id in old_ids
            if old_id not in new_id_set
        ]
        if stale_old_ids:
            try:
                self.vector_store.collection.delete(ids=stale_old_ids)
                logger.info(
                    "ingest_file: deleted %d stale chunks for %s.",
                    len(stale_old_ids),
                    filename,
                )
            except Exception as exc:
                # Do not roll back the successfully stored new version. The
                # manifest is intentionally updated below so the next full
                # ingest can reconcile any leftover stale records.
                logger.warning(
                    "ingest_file: could not delete stale chunks for %s: %s",
                    filename,
                    exc,
                )

        # ------------------------------------------------------------------
        # Update BM25
        # ------------------------------------------------------------------

        if self.bm25 is not None:
            try:
                self.bm25.add_chunks(new_chunks)
                self.bm25.save_index()
            except Exception as exc:
                logger.warning(
                    "ingest_file: BM25 update failed: %s", exc
                )

        # ------------------------------------------------------------------
        # Update manifest
        # ------------------------------------------------------------------

        manifest[filename] = {
            "hash": file_hash,
            "chunk_ids": ids,
        }
        self._save_manifest(manifest)

        count_after = self.vector_store.get_collection_count()

        logger.info(
            "ingest_file: %s → %d new chunks. Collection count: %d.",
            filename,
            len(ids),
            count_after,
        )

        return {
            "filename": filename,
            "chunks_added": len(ids),
            "collection_count": count_after,
            "skipped": False,
        }

    # ==================================================================
    # MANIFEST HELPERS
    # ==================================================================

    def _manifest_path(self) -> "Path":
        return self.vectorstore_path / "ingestion_manifest.json"

    def _load_manifest(self) -> "dict[str, Any]":
        import json as _json
        path = self._manifest_path()
        if path.exists():
            try:
                return _json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_manifest(self, manifest: "dict[str, Any]") -> None:
        import json as _json
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    # ==================================================================
    # METADATA
    # ==================================================================


    @staticmethod
    def _build_chunk_metadata(
        chunk: dict[str, Any]
    ) -> dict[str, Any]:

        # IMPORTANT:
        #
        # Do NOT run canonicalize_query() here.
        #
        # This function is processing document metadata, not user queries.
        # Query normalization belongs in answer_question().

        meta = {
            "filename":
                chunk.get(
                    "filename",
                    "",
                ),

            "chunk_id":
                chunk.get(
                    "chunk_id",
                    0,
                ),

            "source_path":
                chunk.get(
                    "source_path",
                    "",
                ),

            "category":
                chunk.get(
                    "category",
                    "",
                )
                or "",

            "section_heading":
                chunk.get(
                    "section_heading",
                    "",
                )
                or "",

            "section_path":
                str(
                    chunk.get(
                        "section_path",
                        [],
                    )
                ),

            "section_level":
                chunk.get(
                    "section_level",
                    0,
                ),

            "chunk_index":
                chunk.get(
                    "chunk_index",
                    0,
                ),

            "total_section_chunks":
                chunk.get(
                    "total_section_chunks",
                    0,
                ),
        }

        safe_meta = {}

        for key, value in meta.items():

            if value is None:

                safe_meta[key] = ""

            elif isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):

                safe_meta[key] = value

            else:

                safe_meta[key] = str(
                    value
                )

        return safe_meta

    # ==================================================================
    # DENSE RETRIEVAL
    # ==================================================================

    def _dense_retrieve(
        self,
        query_embedding: list[float],
    ) -> dict[str, Candidate]:

        retrieval_k = min(
            max(
                VECTOR_TOP_K,
                self.top_k,
                RERANK_CANDIDATE_LIMIT,
            ),
            MAX_RERANK_CANDIDATES,
        )

        try:

            results = self.vector_store.query(
                query_embedding,
                n_results=retrieval_k,
            )

        except Exception as exc:

            logger.warning(
                "Dense retrieval failed: %s",
                exc,
            )

            return {}

        docs = results.get(
            "documents",
            [],
        )

        metadatas = results.get(
            "metadatas",
            [],
        )

        distances = results.get(
            "distances",
            [],
        )

        docs_list = (
            docs[0]
            if docs
            and isinstance(
                docs[0],
                list,
            )
            else docs
        )

        metas_list = (
            metadatas[0]
            if metadatas
            and isinstance(
                metadatas[0],
                list,
            )
            else metadatas
        )

        dist_list = (
            distances[0]
            if distances
            and isinstance(
                distances[0],
                list,
            )
            else distances
        )

        candidates: dict[
            str,
            Candidate,
        ] = {}

        for rank, document in enumerate(
            docs_list
        ):

            if not document:

                continue

            metadata = (
                metas_list[rank]
                if rank < len(metas_list)
                and metas_list[rank]
                else {}
            )

            distance = (
                dist_list[rank]
                if rank < len(dist_list)
                else None
            )

            filename = str(
                metadata.get(
                    "filename",
                    f"document-{rank}",
                )
            )

            chunk_id = metadata.get(
                "chunk_id",
                rank,
            )

            key = self._candidate_key(
                filename,
                chunk_id,
            )

            candidates[key] = Candidate(

                key=key,

                filename=filename,

                chunk_id=chunk_id,

                text=str(document),

                metadata=metadata,

                dense_rank=rank,

                dense_distance=(
                    self._safe_float(
                        distance
                    )
                ),
            )

        return candidates

    # ==================================================================
    # BM25 RETRIEVAL
    # ==================================================================

    def _bm25_retrieve(
        self,
        search_text: str,
        candidates: dict[str, Candidate],
    ) -> None:

        if self.bm25 is None:

            return

        try:

            retrieval_k = min(
                max(
                    BM25_TOP_K,
                    RERANK_CANDIDATE_LIMIT,
                ),
                MAX_RERANK_CANDIDATES,
            )

            bm25_results = self.bm25.query(
                search_text,
                top_k=retrieval_k,
            )

        except Exception as exc:

            logger.warning(
                "BM25 retrieval failed: %s",
                exc,
            )

            return

        for rank, result in enumerate(
            bm25_results
        ):

            filename = str(
                result.get(
                    "filename",
                    "unknown",
                )
            )

            chunk_id = result.get(
                "chunk_id",
                rank,
            )

            text = (
                result.get(
                    "text",
                    "",
                )
                or ""
            )

            if not text.strip():

                continue

            key = self._candidate_key(
                filename,
                chunk_id,
            )

            if key not in candidates:

                candidates[key] = Candidate(

                    key=key,

                    filename=filename,

                    chunk_id=chunk_id,

                    text=text,

                    metadata={
                        "filename":
                            filename,

                        "chunk_id":
                            chunk_id,

                        "source_path":
                            result.get(
                                "source_path",
                                "",
                            ),

                        "section_heading":
                            result.get(
                                "section_heading",
                                "",
                            ),

                        "section_path":
                            result.get(
                                "section_path",
                                "",
                            ),
                    },
                )

            candidate = candidates[key]

            candidate.bm25_rank = rank

            candidate.bm25_score = (
                self._safe_float(
                    result.get(
                        "bm25_score",
                        0.0,
                    ),
                    default=0.0,
                )
                or 0.0
            )

            # Preserve metadata returned by BM25.
            for field_name in (
                "source_path",
                "section_heading",
                "section_path",
                "section_level",
                "chunk_index",
                "total_section_chunks",
                "category",
            ):

                value = result.get(
                    field_name
                )

                if (
                    value is not None
                    and value != ""
                ):

                    candidate.metadata[
                        field_name
                    ] = value

    # ==================================================================
    # RECIPROCAL RANK FUSION
    # ==================================================================

    @staticmethod
    def _fuse_rrf(
        candidates: dict[str, Candidate],
        rrf_k: int = RRF_K,
    ) -> list[Candidate]:

        """
        Combine dense and BM25 rankings using Reciprocal Rank Fusion.

        RRF does not require dense similarity scores and BM25 scores
        to be numerically comparable.
        """

        for candidate in candidates.values():

            score = 0.0

            if candidate.dense_rank is not None:

                score += (
                    1.0
                    / (
                        rrf_k
                        + candidate.dense_rank
                        + 1
                    )
                )

            if candidate.bm25_rank is not None:

                score += (
                    1.0
                    / (
                        rrf_k
                        + candidate.bm25_rank
                        + 1
                    )
                )

            candidate.rrf_score = score

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: (
                candidate.rrf_score,
                candidate.in_both_retrievers,
            ),
            reverse=True,
        )

        return ranked[
            :RERANK_CANDIDATE_LIMIT
        ]

    # ==================================================================
    # CROSS-ENCODER RERANKING
    # ==================================================================

    def _rerank(
        self,
        question: str,
        candidates: list[Candidate],
    ) -> bool:

        # Reranking is disabled. Never load or invoke a local
        # Hugging Face CrossEncoder.
        if not ENABLE_RERANKER:
            return False

        if (
            self.reranker is None
            or not getattr(
                self.reranker,
                "available",
                False,
            )
            or not candidates
        ):

            return False

        try:

            documents = [
                candidate.text
                for candidate in candidates
                if candidate.text.strip()
            ]

            if len(documents) != len(candidates):

                logger.warning(
                    "Skipping reranking because "
                    "one or more candidates contain "
                    "empty text."
                )

                return False

            scores = self.reranker.rerank(
                question,
                documents,
            )

            if scores is None:

                return False

            scores = list(scores)

            # Do not silently zip mismatched arrays.
            if len(scores) != len(candidates):

                logger.warning(
                    "Reranker returned %d scores "
                    "for %d candidates.",
                    len(scores),
                    len(candidates),
                )

                return False

            for candidate, score in zip(
                candidates,
                scores,
            ):

                parsed_score = (
                    self._safe_float(
                        score
                    )
                )

                if parsed_score is None:

                    logger.warning(
                        "Reranker returned an invalid "
                        "score."
                    )

                    return False

                candidate.rerank_score = (
                    parsed_score
                )

            return True

        except Exception as exc:

            logger.warning(
                "Reranking failed: %s",
                exc,
            )

            return False

    # ==================================================================
    # DEDUPLICATION
    # ==================================================================

    @classmethod
    def _deduplicate_candidates(
        cls,
        candidates: list[Candidate],
    ) -> list[Candidate]:

        seen_keys: set[str] = set()
        seen_text: set[str] = set()

        unique: list[Candidate] = []

        for candidate in candidates:

            if not candidate.text.strip():

                continue

            if candidate.key in seen_keys:

                continue

            normalized_text = cls._normalize_text(
                candidate.text
            )

            if not normalized_text:

                continue

            if normalized_text in seen_text:

                continue

            seen_keys.add(
                candidate.key
            )

            seen_text.add(
                normalized_text
            )

            unique.append(
                candidate
            )

        return unique

    # ==================================================================
    # TOP SCORE PERCENTILE
    # ==================================================================

    @staticmethod
    def _top_score_percentile(
        scores: list[float],
    ) -> float | None:

        """
        Return the percentile position of the maximum score.

        IMPORTANT:

        A broken implementation such as:

            sum(score <= top_score for score in scores)

        always counts every score because top_score is the maximum.

        Here we instead calculate the proportion of scores that are
        strictly below the top score, with tie handling.

        Example:

            [0.1, 0.2, 0.3, 0.4]

        The maximum is clearly at the top.

        If all scores are identical:

            [0.3, 0.3, 0.3, 0.3]

        there is no meaningful separation, so the percentile is treated
        conservatively as 0.5 rather than falsely reporting 1.0.
        """

        if not scores:

            return None

        cleaned = [
            score
            for score in scores
            if math.isfinite(score)
        ]

        if not cleaned:

            return None

        if len(cleaned) == 1:

            return 1.0

        top_score = max(
            cleaned
        )

        strictly_below = sum(
            score < top_score
            for score in cleaned
        )

        equal_to_top = sum(
            score == top_score
            for score in cleaned
        )

        # Mid-rank percentile for ties.
        rank_position = (
            strictly_below
            + (equal_to_top + 1) / 2
        )

        percentile = (
            rank_position
            / len(cleaned)
        )

        return max(
            0.0,
            min(
                1.0,
                percentile,
            ),
        )

    # ==================================================================
    # CONTEXT SELECTION
    # ==================================================================

    def _select_context_candidates(
        self,
        candidates: list[Candidate],
        reranked: bool,
        diversify_sources: bool = False,
    ) -> list[Candidate]:

        if not candidates:

            return []

        # --------------------------------------------------------------
        # SORT FIRST
        # --------------------------------------------------------------

        if reranked:

            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    candidate.rerank_score
                    if candidate.rerank_score
                    is not None
                    else float("-inf")
                ),
                reverse=True,
            )

        else:

            ranked = sorted(
                candidates,
                key=lambda candidate:
                    candidate.rrf_score,
                reverse=True,
            )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Deduplicate BEFORE applying MAX_CONTEXT_CHUNKS.
        #
        # This allows a rank-6 candidate to replace a duplicate/blocked
        # rank-3 candidate instead of losing the opportunity entirely.
        # --------------------------------------------------------------

        ranked = (
            self._deduplicate_candidates(
                ranked
            )
        )

        # For clearly multi-intent questions, preserve the strongest usable
        # candidate from each distinct source before filling the remaining
        # context slots by score. This prevents one source from monopolizing
        # the final context when another source answers a second part of the
        # same question.
        if diversify_sources and ranked:
            diversified: list[Candidate] = []
            seen_sources: set[str] = set()

            for candidate in ranked:
                if candidate.filename in seen_sources:
                    continue
                diversified.append(candidate)
                seen_sources.add(candidate.filename)

            for candidate in ranked:
                if candidate not in diversified:
                    diversified.append(candidate)

            ranked = diversified

        selected: list[Candidate] = []

        source_counts: dict[
            str,
            int,
        ] = {}

        current_chars = 0

        for candidate in ranked:

            source_count = (
                source_counts.get(
                    candidate.filename,
                    0,
                )
            )

            if (
                source_count
                >= MAX_CHUNKS_PER_SOURCE
            ):

                continue

            text = (
                candidate.text
                or ""
            ).strip()

            if not text:

                continue

            text_length = len(text)

            # ----------------------------------------------------------
            # If the first/best candidate itself fits, always preserve it.
            # ----------------------------------------------------------

            if not selected:

                if (
                    text_length
                    <= MAX_CONTEXT_CHARS
                ):

                    selected.append(
                        candidate
                    )

                    source_counts[
                        candidate.filename
                    ] = 1

                    current_chars = (
                        text_length
                    )

                    if (
                        len(selected)
                        >= MAX_CONTEXT_CHUNKS
                    ):

                        break

                    continue

                # If the best chunk is individually larger than the
                # context limit, include a bounded version later rather
                # than silently discarding the best evidence.
                truncated = Candidate(
                    key=candidate.key,
                    filename=candidate.filename,
                    chunk_id=candidate.chunk_id,
                    text=text[
                        :MAX_CONTEXT_CHARS
                    ],
                    metadata=candidate.metadata,
                    dense_rank=candidate.dense_rank,
                    dense_distance=candidate.dense_distance,
                    bm25_rank=candidate.bm25_rank,
                    bm25_score=candidate.bm25_score,
                    rrf_score=candidate.rrf_score,
                    rerank_score=candidate.rerank_score,
                )

                selected.append(
                    truncated
                )

                source_counts[
                    candidate.filename
                ] = 1

                current_chars = (
                    MAX_CONTEXT_CHARS
                )

                break

            # ----------------------------------------------------------
            # NORMAL CONTEXT BUDGET
            # ----------------------------------------------------------

            if (
                current_chars
                + text_length
                > MAX_CONTEXT_CHARS
            ):

                continue

            selected.append(
                candidate
            )

            source_counts[
                candidate.filename
            ] = source_count + 1

            current_chars += text_length

            if (
                len(selected)
                >= MAX_CONTEXT_CHUNKS
            ):

                break

        return selected

    # ==================================================================
    # EVIDENCE EVALUATION
    # ==================================================================

    def _evaluate_evidence(
        self,
        candidates: list[Candidate],
        reranked: bool,
    ) -> tuple[
        list[Candidate],
        Evidence,
    ]:

        if not candidates:

            return [], Evidence(

                level="none",

                should_answer=False,

                top_score=0.0,

                score_source="none",

                supporting_chunks=0,

                top_gap=None,

                agreement=False,

                top_percentile=None,

                retrieval_mode="unknown",

                reason=(
                    "No candidates were retrieved."
                ),
            )

        retrieval_mode = (
            self._retrieval_mode(
                candidates
            )
        )

        # ==============================================================
        # RERANKED PATH
        # ==============================================================

        if reranked:

            ranked = sorted(
                [
                    candidate
                    for candidate in candidates
                    if candidate.rerank_score
                    is not None
                ],
                key=lambda candidate:
                    candidate.rerank_score,
                reverse=True,
            )

            if not ranked:

                return [], Evidence(

                    level="none",

                    should_answer=False,

                    top_score=0.0,

                    score_source="reranker",

                    supporting_chunks=0,

                    top_gap=None,

                    agreement=False,

                    top_percentile=None,

                    retrieval_mode=retrieval_mode,

                    reason=(
                        "The reranker did not produce "
                        "usable scores."
                    ),
                )

            # ----------------------------------------------------------
            # DEDUPLICATE BEFORE EVIDENCE CLASSIFICATION
            # ----------------------------------------------------------

            ranked = (
                self._deduplicate_candidates(
                    ranked
                )
            )

            if not ranked:

                return [], Evidence(

                    level="none",

                    should_answer=False,

                    top_score=0.0,

                    score_source="reranker",

                    supporting_chunks=0,

                    top_gap=None,

                    agreement=False,

                    top_percentile=None,

                    retrieval_mode=retrieval_mode,

                    reason=(
                        "No usable unique evidence "
                        "remained after deduplication."
                    ),
                )

            top = ranked[0]

            top_score = (
                top.rerank_score
                if top.rerank_score
                is not None
                else float("-inf")
            )

            # ----------------------------------------------------------
            # ABSOLUTE FLOOR
            #
            # This is essential.
            #
            # Relative separation alone is not sufficient because even
            # a completely irrelevant query will produce a maximum score
            # among the retrieved candidates.
            # ----------------------------------------------------------

            if (
                not math.isfinite(
                    top_score
                )
                or top_score
                < RERANK_ABSTAIN_FLOOR
            ):

                return [], Evidence(

                    level="none",

                    should_answer=False,

                    top_score=top_score,

                    score_source="reranker",

                    supporting_chunks=0,

                    top_gap=None,

                    agreement=(
                        top.in_both_retrievers
                    ),

                    top_percentile=None,

                    retrieval_mode=retrieval_mode,

                    reason=(
                        "The best reranked candidate "
                        "did not pass the calibrated "
                        "absolute relevance floor."
                    ),
                )

            # ----------------------------------------------------------
            # SCORE DISTRIBUTION
            # ----------------------------------------------------------

            scores = [
                candidate.rerank_score
                for candidate in ranked
                if candidate.rerank_score
                is not None
            ]

            top_percentile = (
                self._top_score_percentile(
                    scores
                )
                if len(scores)
                >= MIN_PERCENTILE_POOL
                else None
            )

            # ----------------------------------------------------------
            # TOP-1 / TOP-2 GAP
            # ----------------------------------------------------------

            top_gap = None

            if (
                len(ranked) > 1
                and ranked[1].rerank_score
                is not None
            ):

                top_gap = (
                    top_score
                    - ranked[1].rerank_score
                )

            # ----------------------------------------------------------
            # SUPPORTING CANDIDATES
            #
            # These are candidates reasonably close to the top score.
            # They are not required to be multiple chunks.
            # ----------------------------------------------------------

            supporting = [

                candidate

                for candidate in ranked

                if (
                    candidate.rerank_score
                    is not None

                    and (
                        top_score
                        - candidate.rerank_score
                    )
                    <= SUPPORT_SCORE_WINDOW
                )
            ]

            supporting_count = len(
                supporting
            )

            agreement = (
                top.in_both_retrievers
            )

            # ----------------------------------------------------------
            # RELATIVE SIGNALS
            # ----------------------------------------------------------

            strong_gap = (
                top_gap is not None
                and top_gap
                >= STRONG_RERANK_GAP
            )

            useful_gap = (
                top_gap is not None
                and top_gap
                >= MIN_RERANK_GAP
            )

            strong_percentile = (
                top_percentile is not None
                and top_percentile
                >= STRONG_TOP_PERCENTILE
            )

            useful_percentile = (
                top_percentile is not None
                and top_percentile
                >= MIN_TOP_PERCENTILE
            )

            # ----------------------------------------------------------
            # STRONG EVIDENCE
            #
            # Strong evidence requires the absolute floor first.
            #
            # Then at least one meaningful relative signal:
            #
            # - strong separation
            # - strong distribution position + agreement
            # - agreement + useful separation
            #
            # This avoids treating every max score as strong evidence.
            # ----------------------------------------------------------

            strong = (

                top_score
                >= RERANK_ABSTAIN_FLOOR

                and (
                    strong_gap

                    or (
                        strong_percentile
                        and agreement
                    )

                    or (
                        agreement
                        and useful_gap
                    )
                )
            )

            if strong:

                return (

                    supporting,

                    Evidence(

                        level="strong",

                        should_answer=True,

                        top_score=top_score,

                        score_source="reranker",

                        supporting_chunks=(
                            supporting_count
                        ),

                        top_gap=top_gap,

                        agreement=agreement,

                        top_percentile=(
                            top_percentile
                        ),

                        retrieval_mode=(
                            retrieval_mode
                        ),

                        reason=(
                            "The top reranked candidate "
                            "passed the absolute relevance "
                            "floor and showed strong relative "
                            "evidence."
                        ),
                    ),
                )

            # ----------------------------------------------------------
            # MODERATE EVIDENCE
            #
            # Moderate evidence still requires the absolute floor.
            #
            # Relative evidence can come from:
            #
            # - retriever agreement
            # - useful top gap
            # - useful percentile
            #
            # A single excellent chunk is allowed.
            # ----------------------------------------------------------

            moderate = (

                top_score
                >= RERANK_ABSTAIN_FLOOR

                and (
                    agreement
                    or useful_gap
                    or useful_percentile
                )
            )

            if moderate:

                return (

                    supporting,

                    Evidence(

                        level="moderate",

                        should_answer=True,

                        top_score=top_score,

                        score_source="reranker",

                        supporting_chunks=(
                            supporting_count
                        ),

                        top_gap=top_gap,

                        agreement=agreement,

                        top_percentile=(
                            top_percentile
                        ),

                        retrieval_mode=(
                            retrieval_mode
                        ),

                        reason=(
                            "The top reranked candidate "
                            "passed the absolute relevance "
                            "floor and had sufficient relative "
                            "retrieval evidence."
                        ),
                    ),
                )

            # ----------------------------------------------------------
            # BORDERLINE / ABSTAIN
            # ----------------------------------------------------------

            return [], Evidence(

                level="none",

                should_answer=False,

                top_score=top_score,

                score_source="reranker",

                supporting_chunks=0,

                top_gap=top_gap,

                agreement=agreement,

                top_percentile=(
                    top_percentile
                ),

                retrieval_mode=retrieval_mode,

                reason=(
                    "The top candidate passed the absolute "
                    "floor but did not show sufficient "
                    "relative evidence to answer reliably."
                ),
            )

        # ==============================================================
        # RRF-ONLY EVIDENCE EVALUATION
        # ==============================================================
        # CrossEncoder reranking is disabled. Evidence is therefore based
        # on RRF rank strength, retriever agreement, and source diversity.

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.rrf_score,
                candidate.in_both_retrievers,
            ),
            reverse=True,
        )

        ranked = self._deduplicate_candidates(ranked)

        if not ranked:
            return [], Evidence(
                level="none",
                should_answer=False,
                top_score=0.0,
                score_source="rrf",
                supporting_chunks=0,
                top_gap=None,
                agreement=False,
                top_percentile=None,
                retrieval_mode=retrieval_mode,
                reason="No usable candidates remained after deduplication.",
            )

        top = ranked[0]
        top_rrf = float(top.rrf_score)
        agreement = bool(top.in_both_retrievers)

        # Absolute floor: do not answer merely because something ranked first.
        if (
            not math.isfinite(top_rrf)
            or top_rrf < MODERATE_RRF_THRESHOLD
        ):
            return [], Evidence(
                level="none",
                should_answer=False,
                top_score=top_rrf,
                score_source="rrf",
                supporting_chunks=0,
                top_gap=None,
                agreement=agreement,
                top_percentile=None,
                retrieval_mode=retrieval_mode,
                reason=(
                    "Hybrid retrieval did not produce sufficiently strong "
                    "rank-based evidence."
                ),
            )

        # Select the context that will actually be sent to the LLM first.
        final = self._select_context_candidates(
            ranked,
            reranked=False,
        )

        if not final:
            return [], Evidence(
                level="none",
                should_answer=False,
                top_score=top_rrf,
                score_source="rrf",
                supporting_chunks=0,
                top_gap=None,
                agreement=agreement,
                top_percentile=None,
                retrieval_mode=retrieval_mode,
                reason=(
                    "No usable context remained after diversity filtering."
                ),
            )

        # Count distinct source documents in the actual final context.
        source_names: set[str] = set()

        for candidate in final:
            metadata = candidate.metadata or {}
            source_name = (
                metadata.get("filename")
                or metadata.get("source")
                or metadata.get("source_path")
                or candidate.filename
                or "unknown"
            )
            source_name = str(source_name).strip()
            if source_name:
                source_names.add(source_name)

        unique_sources = len(source_names)

        # Confidence is deliberately NOT a probability.
        #
        # STRONG  = strong RRF + both retrievers + >=2 source documents
        # MODERATE = meaningful agreement OR source diversity
        # WEAK    = only the minimum RRF floor, without corroboration
        if (
            agreement
            and unique_sources >= 2
            and top_rrf >= STRONG_RRF_THRESHOLD
        ):
            confidence = "strong"

        elif agreement or unique_sources >= 2:
            confidence = "moderate"

        else:
            confidence = "weak"

        # Do not generate an answer from weak evidence.
        if confidence == "weak":
            return [], Evidence(
                level="weak",
                should_answer=False,
                top_score=top_rrf,
                score_source="rrf",
                supporting_chunks=0,
                top_gap=None,
                agreement=agreement,
                top_percentile=None,
                retrieval_mode=retrieval_mode,
                reason=(
                    "Retrieval passed the minimum RRF floor but lacked "
                    "sufficient retriever agreement or source diversity. "
                    "Answer generation was blocked because CrossEncoder "
                    "reranking is disabled."
                ),
            )

        return final, Evidence(
            level=confidence,
            should_answer=True,
            top_score=top_rrf,
            score_source="rrf",
            supporting_chunks=len(final),
            top_gap=None,
            agreement=agreement,
            top_percentile=None,
            retrieval_mode=retrieval_mode,
            reason=(
                "Answer generated from Dense + BM25 hybrid retrieval "
                "using Reciprocal Rank Fusion. CrossEncoder reranking "
                "is disabled."
            ),
        )

    # ==================================================================
    # CONTEXT BUILDING
    # ==================================================================

    @staticmethod
    def _build_context(
        candidates: list[Candidate],
    ) -> tuple[
        str,
        list[str],
        list[dict[str, Any]],
    ]:

        context_parts: list[str] = []

        source_names: list[str] = []

        retrieved_chunks: list[
            dict[str, Any]
        ] = []

        for candidate in candidates:

            section = (
                candidate.metadata.get(
                    "section_heading",
                    "",
                )
                or ""
            )

            context_parts.append(
                f"Source: {candidate.filename}\n"
                f"Section: {section}\n"
                f"{candidate.text}"
            )

            source_names.append(
                candidate.filename
            )

            score = (
                candidate.rerank_score
                if candidate.rerank_score
                is not None
                else candidate.rrf_score
            )

            retrieved_chunks.append({

                "filename":
                    candidate.filename,

                "chunk_id":
                    candidate.chunk_id,

                "text":
                    candidate.text,

                "score":
                    float(score),

                "rerank_score":
                    candidate.rerank_score,

                "rrf_score":
                    candidate.rrf_score,

                "dense_distance":
                    candidate.dense_distance,

                "bm25_score":
                    candidate.bm25_score,

                "dense_rank":
                    candidate.dense_rank,

                "bm25_rank":
                    candidate.bm25_rank,

                "retrieval_agreement":
                    candidate.in_both_retrievers,

                "section_heading":
                    candidate.metadata.get(
                        "section_heading",
                        "",
                    ),

                "section_path":
                    candidate.metadata.get(
                        "section_path",
                        "",
                    ),

                "section_level":
                    candidate.metadata.get(
                        "section_level",
                        0,
                    ),

                "chunk_index":
                    candidate.metadata.get(
                        "chunk_index",
                        0,
                    ),

                "total_section_chunks":
                    candidate.metadata.get(
                        "total_section_chunks",
                        0,
                    ),
            })

        context = (
            "\n\n---\n\n".join(
                context_parts
            )
        )

        unique_sources = list(
            dict.fromkeys(
                source_names
            )
        )

        return (
            context,
            unique_sources,
            retrieved_chunks,
        )

    # ==================================================================
    # LLM
    # ==================================================================

    def _ask_llm(
        self,
        question: str,
        context: str,
    ) -> str:

        api_key = GE_API_KEY or LLM_API_KEY
        model_name = LLM_MODEL
        base_url = LLM_BASE_URL

        if not api_key:

            logger.error(
                "LLM API key is not configured."
            )

            return (
                "I could not generate the answer "
                "because the language-model API "
                "configuration is incomplete."
            )

        if not base_url:

            logger.error(
                "LLM base URL is not configured."
            )

            return (
                "I could not generate the answer "
                "because the language-model endpoint "
                "is not configured."
            )

        if not model_name:

            logger.error(
                "LLM model is not configured."
            )

            return (
                "I could not generate the answer "
                "because the language-model model "
                "is not configured."
            )

        question_lower = (
            question.lower()
        )

        is_explanation = any(
            word in question_lower
            for word in (
                "explain",
                "describe",
                "why",
                "difference",
                "compare",
                "how does",
            )
        )

        is_procedural = any(
            phrase in question_lower
            for phrase in (
                "how to",
                "how do i",
                "steps",
                "procedure",
                "configure",
                "setup",
                "set up",
                "install",
            )
        )

        user_instruction = (
            "Question:\n"
            f"{question}\n\n"

            "Retrieved Information:\n"
            f"{context}\n\n"

            "Answer the question using ONLY "
            "the retrieved information above.\n\n"

            "Do not use outside knowledge.\n\n"

            "If the retrieved information is "
            "insufficient, explicitly say that "
            "the knowledge base does not contain "
            "enough information to answer.\n\n"

            "Only use source filenames that appear "
            "in the retrieved information.\n\n"

            "Include the source filenames under "
            "Sources."
        )

        if is_explanation:

            user_instruction += (
                "\nFor explanations, organize "
                "the answer clearly and use "
                "bullet points where useful."
            )

        if is_procedural:

            user_instruction += (
                "\nFor procedures, use numbered "
                "steps and include prerequisites "
                "only when they are supported by "
                "the retrieved information."
            )

        messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },

            {
                "role": "user",
                "content": user_instruction,
            },
        ]

        try:

            normalized_base_url = (
                base_url.strip()
            )

            if not normalized_base_url.endswith(
                "/"
            ):

                normalized_base_url += "/"

            client = OpenAI(
                api_key=api_key,
                base_url=normalized_base_url,
            )

            response = (
                client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                )
            )

            if not response.choices:

                logger.error(
                    "Language model returned no choices."
                )

                return (
                    "The language model returned "
                    "no response."
                )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            if not content:

                logger.error(
                    "Language model returned empty content."
                )

                return (
                    "The language model returned "
                    "an empty response."
                )

            return content.strip()

        except Exception as exc:

            # Keep technical details in logs rather than exposing
            # endpoint/configuration information to the end user.
            logger.exception(
                "LLM request failed: %s",
                exc,
            )

            return (
                "The language model request failed. "
                "Please check the model configuration "
                "and connectivity."
            )

    # ==================================================================
    # MAIN ENTRY POINT
    # ==================================================================

    def answer_question(
        self,
        question: str,
    ) -> dict[str, Any]:

        def empty_response(
            answer: str
        ) -> dict[str, Any]:

            return {
                "answer": answer,
                "sources": [],
                "retrieved_chunks": [],
                "num_retrieved": 0,
                "evidence": None,
            }

        # --------------------------------------------------------------
        # VALIDATE QUESTION
        # --------------------------------------------------------------

        if (
            not question
            or not question.strip()
        ):

            return empty_response(
                "Please enter a question."
            )

        question = question.strip()

        # --------------------------------------------------------------
        # VALIDATE KB
        # --------------------------------------------------------------

        if not self.knowledge_base_path.exists():

            return empty_response(
                "The knowledge base does not exist. "
                "Add documents and run ingestion."
            )

        # --------------------------------------------------------------
        # VALIDATE VECTORSTORE
        # --------------------------------------------------------------

        try:

            collection_count = (
                self.vector_store
                .get_collection_count()
            )

        except Exception as exc:

            logger.warning(
                "Vector store check failed: %s",
                exc,
            )

            return empty_response(
                "The vector store could not be initialized."
            )

        if collection_count == 0:

            # IMPORTANT:
            #
            # BM25 can still work without the vector store.
            # We therefore do not immediately abort here.
            logger.info(
                "Vector store is empty; "
                "continuing with available lexical retrieval."
            )

        # --------------------------------------------------------------
        # QUERY PREPARATION
        # --------------------------------------------------------------

        search_text = (
            self._prepare_search_query(
                question
            )
        )

        logger.info(
            "Original query: %s",
            question,
        )

        logger.info(
            "Search query: %s",
            search_text,
        )

        # --------------------------------------------------------------
        # QUERY EMBEDDING
        # --------------------------------------------------------------

        query_embedding = None

        if collection_count > 0:

            try:

                query_embedding = (
                    self.embedding_service
                    .create_embedding(
                        search_text
                    )
                )

            except Exception as exc:

                logger.warning(
                    "Query embedding failed: %s",
                    exc,
                )

        # --------------------------------------------------------------
        # DENSE RETRIEVAL
        # --------------------------------------------------------------

        candidates: dict[
            str,
            Candidate,
        ] = {}

        if query_embedding:

            candidates = (
                self._dense_retrieve(
                    query_embedding
                )
            )

        # --------------------------------------------------------------
        # BM25 RETRIEVAL
        # --------------------------------------------------------------

        # BM25 is attempted regardless of embedding success.
        #
        # This is important because a temporary embedding/API problem
        # should not destroy lexical retrieval.
        self._bm25_retrieve(
            search_text,
            candidates,
        )

        # --------------------------------------------------------------
        # NO RETRIEVAL RESULTS
        # --------------------------------------------------------------

        if not candidates:

            return (
                self._answer_via_keyword_fallback(
                    question,
                    search_text,
                )
            )

        # --------------------------------------------------------------
        # RRF FUSION
        # --------------------------------------------------------------

        fused = self._fuse_rrf(
            candidates
        )

        if not fused:

            return (
                self._answer_via_keyword_fallback(
                    question,
                    search_text,
                )
            )

        logger.info(
            "Hybrid candidate pool: %d",
            len(fused),
        )

        # --------------------------------------------------------------
        # CROSS-ENCODER RERANKING
        # --------------------------------------------------------------

        reranked = self._rerank(
            question,
            fused,
        )

        logger.info(
            "Reranking successful: %s",
            reranked,
        )

        # --------------------------------------------------------------
        # EVIDENCE GATE
        # --------------------------------------------------------------

        final_candidates, evidence = (
            self._evaluate_evidence(
                fused,
                reranked,
            )
        )

        if not evidence.should_answer:

            return {

                "answer":
                    ABSTAIN_MESSAGE,

                "sources": [],

                "retrieved_chunks": [],

                "num_retrieved": 0,

                "evidence":
                    evidence.as_dict(),
            }

        # --------------------------------------------------------------
        # FINAL CONTEXT
        #
        # This is intentionally applied AFTER evidence evaluation but
        # BEFORE final context truncation.
        #
        # Deduplication happens inside _select_context_candidates()
        # BEFORE MAX_CONTEXT_CHUNKS is enforced.
        # --------------------------------------------------------------

        final_candidates = (
            self._select_context_candidates(
                final_candidates,
                reranked=reranked,
                diversify_sources=self._is_multi_intent_query(
                    question
                ),
            )
        )

        if not final_candidates:

            fallback_evidence = Evidence(

                level="none",

                should_answer=False,

                top_score=0.0,

                score_source="none",

                supporting_chunks=0,

                top_gap=None,

                agreement=False,

                top_percentile=None,

                retrieval_mode=(
                    evidence.retrieval_mode
                ),

                reason=(
                    "No usable evidence remained "
                    "after context filtering."
                ),
            )

            return {

                "answer":
                    ABSTAIN_MESSAGE,

                "sources": [],

                "retrieved_chunks": [],

                "num_retrieved": 0,

                "evidence":
                    fallback_evidence.as_dict(),
            }

        # --------------------------------------------------------------
        # BUILD CONTEXT
        # --------------------------------------------------------------

        (
            context,
            source_names,
            retrieved_chunks,
        ) = self._build_context(
            final_candidates
        )

        # --------------------------------------------------------------
        # GENERATE ANSWER
        # --------------------------------------------------------------

        answer = self._ask_llm(
            question,
            context,
        )

        answer = (
            f"{answer}\n\n"
            f"Confidence: "
            f"{evidence.level.title()} evidence"
        )

        return {

            "answer":
                answer,

            "sources":
                source_names,

            "retrieved_chunks":
                retrieved_chunks,

            "num_retrieved":
                len(retrieved_chunks),

            "evidence":
                evidence.as_dict(),
        }

    # ==================================================================
    # FALLBACK
    # ==================================================================

    def _answer_via_keyword_fallback(
        self,
        question: str,
        search_text: str,
    ) -> dict[str, Any]:

        # --------------------------------------------------------------
        # FIRST TRY BM25
        #
        # BM25 is already indexed and is much cheaper than loading and
        # rechunking the entire KB for every failed query.
        # --------------------------------------------------------------

        if self.bm25 is not None:

            try:

                results = self.bm25.query(
                    search_text,
                    top_k=MAX_CONTEXT_CHUNKS,
                )

                documents = []

                for result in results:

                    text = (
                        result.get(
                            "text",
                            "",
                        )
                        or ""
                    )

                    if not text.strip():

                        continue

                    documents.append({

                        "filename":
                            result.get(
                                "filename",
                                "unknown",
                            ),

                        "text":
                            text,

                        "chunk_id":
                            result.get(
                                "chunk_id",
                                0,
                            ),

                        "score":
                            float(
                                result.get(
                                    "bm25_score",
                                    0.0,
                                )
                                or 0.0
                            ),

                        "section_heading":
                            result.get(
                                "section_heading",
                                "",
                            ),

                        "section_path":
                            result.get(
                                "section_path",
                                "",
                            ),

                        "section_level":
                            result.get(
                                "section_level",
                                0,
                            ),

                        "chunk_index":
                            result.get(
                                "chunk_index",
                                0,
                            ),
                    })

                if documents:

                    return self._answer_from_documents(
                        question,
                        documents,
                        retrieval_mode="bm25_fallback",
                    )

            except Exception as exc:

                logger.warning(
                    "BM25 fallback failed: %s",
                    exc,
                )

        # --------------------------------------------------------------
        # LAST-RESORT LEXICAL SCAN
        #
        # This is intentionally only used when BM25 is unavailable or
        # failed. It is not part of the normal retrieval path.
        # --------------------------------------------------------------

        keyword_docs = self._keyword_search(
            [search_text]
        )

        if not keyword_docs:

            return {

                "answer":
                    ABSTAIN_MESSAGE,

                "sources": [],

                "retrieved_chunks": [],

                "num_retrieved": 0,

                "evidence":
                    Evidence(

                        level="none",

                        should_answer=False,

                        top_score=0.0,

                        score_source="keyword",

                        supporting_chunks=0,

                        top_gap=None,

                        agreement=False,

                        top_percentile=None,

                        retrieval_mode=(
                            "keyword_fallback"
                        ),

                        reason=(
                            "No relevant information "
                            "was found in the knowledge base."
                        ),
                    ).as_dict(),
            }

        return self._answer_from_documents(
            question,
            keyword_docs,
            retrieval_mode="keyword_fallback",
        )

    # ==================================================================
    # ANSWER FROM FALLBACK DOCUMENTS
    # ==================================================================

    def _answer_from_documents(
        self,
        question: str,
        documents: list[
            dict[str, Any]
        ],
        retrieval_mode: str = "keyword_fallback",
    ) -> dict[str, Any]:

        # --------------------------------------------------------------
        # Deduplicate fallback documents.
        # --------------------------------------------------------------

        unique_documents = []

        seen_keys: set[str] = set()
        seen_text: set[str] = set()

        for document in documents:

            text = (
                document.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            if not text:

                continue

            filename = str(
                document.get(
                    "filename",
                    "unknown",
                )
            )

            chunk_id = document.get(
                "chunk_id",
                0,
            )

            key = self._candidate_key(
                filename,
                chunk_id,
            )

            normalized_text = (
                self._normalize_text(
                    text
                )
            )

            if key in seen_keys:

                continue

            if normalized_text in seen_text:

                continue

            seen_keys.add(key)
            seen_text.add(
                normalized_text
            )

            unique_documents.append(
                document
            )

        unique_documents = (
            unique_documents[
                :MAX_CONTEXT_CHUNKS
            ]
        )

        if not unique_documents:

            return {

                "answer":
                    ABSTAIN_MESSAGE,

                "sources": [],

                "retrieved_chunks": [],

                "num_retrieved": 0,

                "evidence":
                    Evidence(

                        level="none",

                        should_answer=False,

                        top_score=0.0,

                        score_source="fallback",

                        supporting_chunks=0,

                        top_gap=None,

                        agreement=False,

                        top_percentile=None,

                        retrieval_mode=(
                            retrieval_mode
                        ),

                        reason=(
                            "No usable fallback "
                            "documents were found."
                        ),
                    ).as_dict(),
            }

        # --------------------------------------------------------------
        # Source diversity
        # --------------------------------------------------------------

        selected_documents = []

        source_counts: dict[
            str,
            int,
        ] = {}

        current_chars = 0

        for document in unique_documents:

            filename = str(
                document.get(
                    "filename",
                    "unknown",
                )
            )

            count = source_counts.get(
                filename,
                0,
            )

            if (
                count
                >= MAX_CHUNKS_PER_SOURCE
            ):

                continue

            text = (
                document.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            if not text:

                continue

            if not selected_documents:

                if len(text) > MAX_CONTEXT_CHARS:

                    document = dict(
                        document
                    )

                    document["text"] = (
                        text[
                            :MAX_CONTEXT_CHARS
                        ]
                    )

                    text = document["text"]

                selected_documents.append(
                    document
                )

                source_counts[
                    filename
                ] = 1

                current_chars = len(text)

                continue

            if (
                current_chars
                + len(text)
                > MAX_CONTEXT_CHARS
            ):

                continue

            selected_documents.append(
                document
            )

            source_counts[
                filename
            ] = count + 1

            current_chars += len(text)

        if not selected_documents:

            return {

                "answer":
                    ABSTAIN_MESSAGE,

                "sources": [],

                "retrieved_chunks": [],

                "num_retrieved": 0,

                "evidence":
                    Evidence(

                        level="none",

                        should_answer=False,

                        top_score=0.0,

                        score_source="fallback",

                        supporting_chunks=0,

                        top_gap=None,

                        agreement=False,

                        top_percentile=None,

                        retrieval_mode=(
                            retrieval_mode
                        ),

                        reason=(
                            "No fallback context "
                            "fit the context limits."
                        ),
                    ).as_dict(),
            }

        context = (
            "\n\n---\n\n".join(
                (
                    f"Source: "
                    f"{doc.get('filename', 'unknown')}\n"
                    f"Section: "
                    f"{doc.get('section_heading', '')}\n"
                    f"{doc.get('text', '')}"
                )
                for doc in selected_documents
            )
        )

        sources = list(
            dict.fromkeys(
                str(
                    doc.get(
                        "filename",
                        "unknown",
                    )
                )
                for doc in selected_documents
            )
        )

        retrieved_chunks = [
            {
                "filename":
                    doc.get(
                        "filename",
                        "unknown",
                    ),

                "chunk_id":
                    doc.get(
                        "chunk_id",
                        0,
                    ),

                "text":
                    doc.get(
                        "text",
                        "",
                    ),

                "score":
                    float(
                        doc.get(
                            "score",
                            0.0,
                        )
                        or 0.0
                    ),

                "section_heading":
                    doc.get(
                        "section_heading",
                        "",
                    ),

                "section_path":
                    doc.get(
                        "section_path",
                        "",
                    ),

                "section_level":
                    doc.get(
                        "section_level",
                        0,
                    ),

                "chunk_index":
                    doc.get(
                        "chunk_index",
                        0,
                    ),
            }
            for doc in selected_documents
        ]

        answer = self._ask_llm(
            question,
            context,
        )

        answer += (
            "\n\n"
            "Confidence: Weak evidence "
            f"({retrieval_mode})"
        )

        top_score = float(
            selected_documents[0].get(
                "score",
                0.0,
            )
            or 0.0
        )

        return {

            "answer":
                answer,

            "sources":
                sources,

            "retrieved_chunks":
                retrieved_chunks,

            "num_retrieved":
                len(
                    retrieved_chunks
                ),

            "evidence":
                Evidence(

                    level="weak",

                    should_answer=True,

                    top_score=top_score,

                    score_source="fallback",

                    supporting_chunks=len(
                        selected_documents
                    ),

                    top_gap=None,

                    agreement=False,

                    top_percentile=None,

                    retrieval_mode=(
                        retrieval_mode
                    ),

                    reason=(
                        "Answer generated using "
                        "a fallback lexical retriever."
                    ),
                ).as_dict(),
        }

    # ==================================================================
    # KEYWORD SEARCH
    # ==================================================================

    def _keyword_search(
        self,
        terms: list[str],
        top_k: int = 5,
    ) -> list[
        dict[str, Any]
    ]:

        try:

            docs = load_documents(
                self.knowledge_base_path
            )

            chunks = chunk_documents(
                docs,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        except Exception as exc:

            logger.warning(
                "Keyword search failed: %s",
                exc,
            )

            return []

        # --------------------------------------------------------------
        # NORMALIZE TERMS
        # --------------------------------------------------------------

        normalized_terms = []

        for term in terms:

            if not term:

                continue

            cleaned = re.sub(
                r"\s+",
                " ",
                term.lower().strip(),
            )

            if cleaned:

                normalized_terms.append(
                    cleaned
                )

        if not normalized_terms:

            return []

        # --------------------------------------------------------------
        # TOKENIZE
        # --------------------------------------------------------------

        tokens = set()

        for term in normalized_terms:

            extracted = re.findall(
                r"\b[a-zA-Z0-9][a-zA-Z0-9_-]{2,}\b",
                term,
            )

            tokens.update(
                extracted
            )

        if not tokens:

            return []

        # --------------------------------------------------------------
        # SCORE CHUNKS
        # --------------------------------------------------------------

        scored = []

        for chunk in chunks:

            text = (
                chunk.get(
                    "text",
                    "",
                )
                or ""
            ).lower()

            if not text:

                continue

            # ----------------------------------------------------------
            # Exact phrase matches
            # ----------------------------------------------------------

            phrase_score = sum(

                text.count(term) * 3

                for term in normalized_terms

                if term in text
            )

            # ----------------------------------------------------------
            # Token overlap
            # ----------------------------------------------------------

            token_score = sum(

                text.count(token)

                for token in tokens

                if token in text
            )

            total_score = (
                phrase_score
                + token_score
            )

            # ----------------------------------------------------------
            # Require actual lexical overlap.
            # ----------------------------------------------------------

            if total_score <= 0:

                continue

            scored.append(
                (
                    total_score,

                    {
                        "filename":
                            chunk.get(
                                "filename",
                                "unknown",
                            ),

                        "text":
                            chunk.get(
                                "text",
                                "",
                            ),

                        "chunk_id":
                            chunk.get(
                                "chunk_id",
                                0,
                            ),

                        "score":
                            float(
                                total_score
                            ),

                        "section_heading":
                            chunk.get(
                                "section_heading",
                                "",
                            ),

                        "section_path":
                            chunk.get(
                                "section_path",
                                "",
                            ),

                        "section_level":
                            chunk.get(
                                "section_level",
                                0,
                            ),

                        "chunk_index":
                            chunk.get(
                                "chunk_index",
                                0,
                            ),
                    },
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # --------------------------------------------------------------
        # Deduplicate lexical fallback results.
        # --------------------------------------------------------------

        results = []

        seen_keys: set[str] = set()
        seen_text: set[str] = set()

        for score, document in scored:

            filename = str(
                document.get(
                    "filename",
                    "unknown",
                )
            )

            chunk_id = document.get(
                "chunk_id",
                0,
            )

            key = self._candidate_key(
                filename,
                chunk_id,
            )

            normalized_text = (
                self._normalize_text(
                    document.get(
                        "text",
                        "",
                    )
                )
            )

            if key in seen_keys:

                continue

            if normalized_text in seen_text:

                continue

            seen_keys.add(key)
            seen_text.add(
                normalized_text
            )

            results.append(
                document
            )

            if len(results) >= top_k:

                break

        return results
