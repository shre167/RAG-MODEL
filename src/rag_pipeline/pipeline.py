from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.config import (
    BM25_TOP_K,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    GE_API_KEY,
    HYBRID_TOP_K,
    KNOWLEDGE_BASE_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    RERANK_TOP_K,
    SCORE_THRESHOLD,
    TOP_K,
    VECTOR_TOP_K,
    VECTORSTORE_DIR,
)
from src.document_loader import (
    chunk_documents,
    list_txt_files,
    load_documents,
)
from src.embeddings import EmbeddingService
from src.bm25_retriever import BM25Retriever
from src.utils import ensure_directory
from src.vector_store import ChromaVectorStore

from src.rag_pipeline.config import (
    ABSTAIN_MESSAGE,
    ENABLE_RERANKER,
    MAX_CHUNKS_PER_SOURCE,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
    MAX_RERANK_CANDIDATES,
    RERANK_CANDIDATE_LIMIT,
    RRF_K,
)
from src.rag_pipeline.context import (
    build_context,
    deduplicate_candidates,
    select_context_candidates,
)
from src.rag_pipeline.evidence import (
    evaluate_evidence,
    top_score_percentile,
)
from src.rag_pipeline.fallback import (
    answer_from_documents,
    answer_via_keyword_fallback,
    keyword_search,
)
from src.rag_pipeline.generation import ask_llm
from src.rag_pipeline.models import Candidate, Evidence
from src.rag_pipeline.query import (
    candidate_key,
    get_retrieval_mode,
    is_greeting,
    is_multi_intent_query,
    normalize_text,
    prepare_search_query,
    safe_float,
)
from src.rag_pipeline.evaluation import evaluate_retrieval, evaluate_answer_quality
from src.rag_pipeline.observability import observe_pipeline_answer
from src.rag_pipeline.observation_store import record_observation
from src.rag_pipeline.citations import generate_claim_citations
from src.rag_pipeline.confidence import evaluate_confidence
from src.rag_pipeline.kb_state import (
    check_index_consistency,
    delete_document as _delete_kb_document,
    get_current_kb_state,
    increment_kb_version,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Domain-independent RAG pipeline.

    Architecture:
        User Query
             ↓
        Greeting Check
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
            Candidate Pool
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

    CrossEncoder reranking is intentionally disabled.
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
        self.knowledge_base_path = Path(knowledge_base_path)
        self.vectorstore_path = ensure_directory(vectorstore_path)
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # --------------------------------------------------------------
        # Embedding Service
        # --------------------------------------------------------------
        if os.getenv(
            "DEV_EMBEDDINGS",
            "false",
        ).lower() in ("1", "true", "yes"):
            from src.embeddings import SyntheticEmbeddingService

            self.embedding_service = SyntheticEmbeddingService()
        else:
            self.embedding_service = EmbeddingService(
                api_key=GE_API_KEY or LLM_API_KEY,
                model=model_name,
                base_url=LLM_BASE_URL,
            )

        # --------------------------------------------------------------
        # Vector Store
        # --------------------------------------------------------------
        self.vector_store = ChromaVectorStore(
            collection_name="knowledge_base",
            persist_directory=self.vectorstore_path,
        )

        try:
            self.index_loaded = (
                self.vector_store.get_collection_count() > 0
            )
        except Exception:
            self.index_loaded = False

        # --------------------------------------------------------------
        # BM25
        # --------------------------------------------------------------
        self.bm25 = None

        try:
            self.bm25 = BM25Retriever(
                kb_path=str(self.knowledge_base_path),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            if self.knowledge_base_path.exists():
                self.bm25.build_index(
                    str(self.knowledge_base_path)
                )

            logger.info(
                "BM25 retriever initialized."
            )

        except Exception as exc:
            logger.warning(
                "BM25 initialization failed: %s",
                exc,
            )

        # --------------------------------------------------------------
        # CrossEncoder intentionally disabled
        # --------------------------------------------------------------
        self.reranker = None

        logger.info(
            "CrossEncoder reranker disabled; "
            "using Dense + BM25 + RRF only."
        )

    # ==================================================================
    # DELEGATING HELPERS
    # ==================================================================

    @staticmethod
    def _candidate_key(
        filename: str,
        chunk_id: Any,
    ) -> str:
        return candidate_key(
            filename,
            chunk_id,
        )

    @staticmethod
    def _safe_float(
        value: Any,
        default: float | None = None,
    ) -> float | None:
        return safe_float(
            value,
            default,
        )

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        return normalize_text(text)

    def _retrieval_mode(
        self,
        candidates: list[Candidate],
    ) -> str:
        return get_retrieval_mode(
            candidates
        )

    @staticmethod
    def _is_multi_intent_query(
        question: str,
    ) -> bool:
        return is_multi_intent_query(
            question
        )

    @staticmethod
    def _prepare_search_query(
        question: str,
    ) -> str:
        result = prepare_search_query(question)

        if isinstance(result, dict):
            return str(
                result.get(
                    "search_query",
                    result.get(
                        "normalized_query",
                        question,
                    ),
                )
            )

        return str(result)

    def _dense_retrieve(
        self,
        query_embedding: list[float],
    ) -> dict[str, Candidate]:
        from src.rag_pipeline.retrieval import (
            dense_retrieve,
        )

        return dense_retrieve(
            self.vector_store,
            query_embedding,
            self.top_k,
        )

    def _bm25_retrieve(
        self,
        search_text: str,
        candidates: dict[str, Candidate],
    ) -> None:
        from src.rag_pipeline.retrieval import (
            bm25_retrieve,
        )

        bm25_retrieve(
            self.bm25,
            search_text,
            candidates,
        )

    @staticmethod
    def _fuse_rrf(
        candidates: dict[str, Candidate],
        rrf_k: int = RRF_K,
    ) -> list[Candidate]:
        from src.rag_pipeline.retrieval import (
            fuse_rrf,
        )

        return fuse_rrf(
            candidates,
            rrf_k=rrf_k,
        )

    def _rerank(
        self,
        question: str,
        candidates: list[Candidate],
    ) -> bool:
        from src.rag_pipeline.retrieval import (
            rerank,
        )

        return rerank(
            question,
            candidates,
            self.reranker,
        )

    @classmethod
    def _deduplicate_candidates(
        cls,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        return deduplicate_candidates(
            candidates
        )

    @staticmethod
    def _top_score_percentile(
        scores: list[float],
    ) -> float | None:
        return top_score_percentile(
            scores
        )

    def _select_context_candidates(
        self,
        candidates: list[Candidate],
        reranked: bool,
        diversify_sources: bool = False,
        retrieval_mode: str | None = None,
    ) -> list[Candidate]:
        return select_context_candidates(
            candidates,
            reranked=reranked,
            diversify_sources=diversify_sources,
            retrieval_mode=retrieval_mode,
        )

    def _evaluate_evidence(
        self,
        candidates: list[Candidate],
        reranked: bool,
        search_text: str,
        retrieval_mode: str | None = None,
    ) -> tuple[list[Candidate], Evidence]:
        return evaluate_evidence(
            candidates,
            reranked,
            search_text,
            retrieval_mode=retrieval_mode,
        )

    @staticmethod
    def _build_context(
        candidates: list[Candidate],
    ) -> tuple[
        str,
        list[str],
        list[dict[str, Any]],
    ]:
        return build_context(
            candidates
        )

    def _ask_llm(
        self,
        question: str,
        context: str,
        trace: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate an answer.

        The optional trace dictionary is passed through to the generation
        layer for Observatory instrumentation. Existing callers remain
        compatible because trace is optional.
        """
        return ask_llm(
            question,
            context,
            trace=trace,
        )

    def _answer_via_keyword_fallback(
        self,
        question: str,
        search_text: str,
    ) -> dict[str, Any]:
        return answer_via_keyword_fallback(
            question,
            search_text,
            bm25=self.bm25,
            knowledge_base_path=self.knowledge_base_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            ask_llm_fn=self._ask_llm,
        )

    def _answer_from_documents(
        self,
        question: str,
        documents: list[dict[str, Any]],
        retrieval_mode: str = "keyword_fallback",
    ) -> dict[str, Any]:
        return answer_from_documents(
            question,
            documents,
            ask_llm_fn=self._ask_llm,
            retrieval_mode=retrieval_mode,
        )

    def _keyword_search(
        self,
        terms: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return keyword_search(
            terms,
            knowledge_base_path=self.knowledge_base_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            top_k=top_k,
        )

    # ==================================================================
    # STATUS
    # ==================================================================

    def status(self) -> dict[str, Any]:
        files = list_txt_files(
            self.knowledge_base_path
        )

        try:
            collection_count = (
                self.vector_store.get_collection_count()
            )
        except Exception:
            collection_count = 0

        return {
            "knowledge_base_exists": (
                self.knowledge_base_path.exists()
            ),
            "txt_files_found": len(files),
            "index_available": collection_count > 0,
            "collection_count": collection_count,
            "vectorstore_path": str(
                self.vectorstore_path
            ),
            "embedding_model": getattr(
                self.embedding_service,
                "model",
                None,
            ),
            "reranker_available": False,
            "bm25_available": (
                self.bm25 is not None
            ),
            "retrieval_settings": {
                "vector_top_k": VECTOR_TOP_K,
                "bm25_top_k": BM25_TOP_K,
                "hybrid_top_k": HYBRID_TOP_K,
                "rerank_candidate_limit": (
                    RERANK_CANDIDATE_LIMIT
                ),
                "rerank_top_k": RERANK_TOP_K,
                "max_context_chunks": (
                    MAX_CONTEXT_CHUNKS
                ),
                "max_chunks_per_source": (
                    MAX_CHUNKS_PER_SOURCE
                ),
                "max_context_chars": (
                    MAX_CONTEXT_CHARS
                ),
            },
        }

    def answer_question(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
        evaluation_labels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the unchanged pipeline and persist one complete observation."""
        import time

        started = time.perf_counter()
        response = self._answer_question_impl(question, retrieval_mode)
        total_latency_ms = (time.perf_counter() - started) * 1000
        observation = record_observation(
            self.vectorstore_path,
            question,
            response,
            getattr(self.embedding_service, "model", None),
            total_latency_ms,
        )
        response["observation"] = {
            key: observation[key]
            for key in ("query_id", "timestamp", "latency_ms", "embedding_model")
        }
        return observe_pipeline_answer(
            pipeline=self,
            question=question,
            retrieval_mode=retrieval_mode,
            response=response,
            total_latency_ms=total_latency_ms,
            labels=evaluation_labels,
        )

    # ==================================================================
    # INGESTION
    # ==================================================================

    def ingest_documents(
        self,
    ) -> dict[str, Any]:
        """
        Incrementally ingest all documents.

        Existing chunk IDs are preserved. Only missing chunks are
        embedded and added to Chroma.
        """

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
                "No .txt files were found in the knowledge base."
            )

        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = [
            chunk
            for chunk in chunks
            if (
                chunk.get("text", "") or ""
            ).strip()
        ]

        if not chunks:
            raise ValueError(
                "No usable content was found in the knowledge base "
                "after chunking."
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

        ids = [
            self._candidate_key(
                metadata["filename"],
                metadata["chunk_id"],
            )
            for metadata in metadatas
        ]

        if len(set(ids)) != len(ids):
            raise ValueError(
                "Duplicate chunk IDs were generated during ingestion."
            )

        if len(ids) != len(texts):
            raise ValueError(
                f"ID count mismatch: expected "
                f"{len(texts)}, got {len(ids)}"
            )

        count_before = (
            self.vector_store.get_collection_count()
        )

        existing_ids = (
            self.vector_store.get_existing_ids(
                ids
            )
        )

        missing_indices = [
            index
            for index, chunk_id in enumerate(ids)
            if chunk_id not in existing_ids
        ]

        skipped_count = (
            len(ids)
            - len(missing_indices)
        )

        batch_size = max(
            1,
            EMBEDDING_BATCH_SIZE,
        )

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

        for batch_start in range(
            0,
            len(missing_indices),
            batch_size,
        ):
            batch_indices = missing_indices[
                batch_start:
                batch_start + batch_size
            ]

            batch_texts = [
                texts[index]
                for index in batch_indices
            ]

            batch_metadatas = [
                metadatas[index]
                for index in batch_indices
            ]

            batch_ids = [
                ids[index]
                for index in batch_indices
            ]

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
                    f"after persisting "
                    f"{embedded_count}/{len(missing_indices)} "
                    f"missing chunks: {exc}"
                ) from exc

            if (
                not embeddings
                or len(embeddings)
                != len(batch_texts)
            ):
                raise ValueError(
                    f"Embedding count mismatch: expected "
                    f"{len(batch_texts)}, got "
                    f"{len(embeddings) if embeddings else 0}"
                )

            self.vector_store.add_documents(
                texts=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

            embedded_count += len(
                batch_ids
            )

        self.index_loaded = True

        # Rebuild BM25 from the complete knowledge base so that
        # incremental Chroma ingestion and lexical retrieval remain
        # synchronized.
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
            self.vector_store.get_collection_count()
        )

        if count_after == 0:
            raise RuntimeError(
                "Vector store is empty after ingestion."
            )

        new_kb_state = increment_kb_version(self)

        return {
            "chunks_indexed": embedded_count,
            "chunks_skipped": skipped_count,
            "chunks_total": len(ids),
            "files_processed": len(docs),
            "chunks_created": len(chunks),
            "collection_count": count_after,
            "kb_version": new_kb_state.version,
            "indexes_consistent": new_kb_state.indexes_consistent,
        }

    def ingest_file(
        self,
        file_path: str | Path,
        content: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest one file incrementally.

        SHA-256 is used to determine whether the file changed.
        Unchanged files are skipped.

        Changed files:
            1. Read
            2. Hash
            3. Chunk
            4. Embed
            5. Add new chunks
            6. Delete stale chunks
            7. Update BM25
            8. Update manifest
        """

        if isinstance(
            file_path,
            str,
        ):
            file_path = Path(
                file_path
            )

        # --------------------------------------------------------------
        # Resolve target path
        # --------------------------------------------------------------

        if content is not None:
            target = (
                self.knowledge_base_path
                / file_path.name
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            target.write_text(
                content,
                encoding="utf-8",
            )

            file_path = target

        else:
            if not file_path.is_absolute():
                file_path = (
                    self.knowledge_base_path
                    / file_path
                )

            if not file_path.exists():
                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )

            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        filename = file_path.name

        # --------------------------------------------------------------
        # Hash
        # --------------------------------------------------------------

        file_hash = hashlib.sha256(
            content.encode(
                "utf-8"
            )
        ).hexdigest()

        manifest = self._load_manifest()

        existing = manifest.get(
            filename,
            {},
        )

        # --------------------------------------------------------------
        # Unchanged file → skip
        # --------------------------------------------------------------

        if (
            existing.get("hash")
            == file_hash
        ):
            logger.info(
                "ingest_file: %s unchanged "
                "(hash match), skipping.",
                filename,
            )

            return {
                "filename": filename,
                "chunks_added": 0,
                "collection_count": (
                    self.vector_store
                    .get_collection_count()
                ),
                "skipped": True,
                "hash": file_hash,
            }

        # --------------------------------------------------------------
        # Clean + chunk
        # --------------------------------------------------------------

        from src.document_loader import (
            clean_text as _clean_text,
        )

        doc = {
            "filename": filename,
            "source_path": str(
                file_path
            ),
            "content": _clean_text(
                content
            ),
        }

        new_chunks = chunk_documents(
            [doc],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        new_chunks = [
            chunk
            for chunk in new_chunks
            if (
                chunk.get("text")
                or ""
            ).strip()
        ]

        if not new_chunks:
            raise ValueError(
                f"No usable content found in "
                f"{filename} after chunking."
            )

        texts = [
            chunk["text"]
            for chunk in new_chunks
        ]

        metadatas = [
            self._build_chunk_metadata(
                chunk
            )
            for chunk in new_chunks
        ]

        ids = [
            self._candidate_key(
                metadata["filename"],
                metadata["chunk_id"],
            )
            for metadata in metadatas
        ]

        # --------------------------------------------------------------
        # Existing IDs belonging to the previous version
        # --------------------------------------------------------------

        old_ids = list(
            existing.get(
                "chunk_ids",
                [],
            )
        )

        old_id_set = set(
            old_ids
        )

        new_id_set = set(
            ids
        )

        # --------------------------------------------------------------
        # If a changed file reuses a chunk ID, remove the old record
        # before inserting the replacement.
        # --------------------------------------------------------------

        reused_ids = list(
            old_id_set
            & new_id_set
        )

        if reused_ids:
            try:
                self.vector_store.collection.delete(
                    ids=reused_ids
                )

                logger.info(
                    "ingest_file: removed %d "
                    "reused chunk IDs before replacement "
                    "for %s.",
                    len(reused_ids),
                    filename,
                )

            except Exception as exc:
                raise RuntimeError(
                    "Could not replace existing chunk IDs "
                    f"for {filename}: {exc}"
                ) from exc

        # --------------------------------------------------------------
        # Embed new version
        # --------------------------------------------------------------

        try:
            embeddings = (
                self.embedding_service
                .create_embeddings_batch(
                    texts
                )
            )
        except Exception as exc:
            raise ValueError(
                f"Embedding failed for "
                f"{filename}: {exc}"
            ) from exc

        if (
            not embeddings
            or len(embeddings)
            != len(texts)
        ):
            raise ValueError(
                f"Embedding count mismatch for "
                f"{filename}: expected "
                f"{len(texts)}, got "
                f"{len(embeddings) if embeddings else 0}"
            )

        # --------------------------------------------------------------
        # Add new version
        # --------------------------------------------------------------

        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        self.index_loaded = True

        # --------------------------------------------------------------
        # Delete stale old chunks
        # --------------------------------------------------------------

        stale_old_ids = [
            old_id
            for old_id in old_ids
            if old_id not in new_id_set
        ]

        if stale_old_ids:
            try:
                self.vector_store.collection.delete(
                    ids=stale_old_ids
                )

                logger.info(
                    "ingest_file: deleted %d stale "
                    "chunks for %s.",
                    len(stale_old_ids),
                    filename,
                )

            except Exception as exc:
                logger.warning(
                    "ingest_file: could not delete "
                    "stale chunks for %s: %s",
                    filename,
                    exc,
                )

        # --------------------------------------------------------------
        # BM25 synchronization
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
                    "ingest_file: BM25 update failed: %s",
                    exc,
                )

        # --------------------------------------------------------------
        # Manifest
        # --------------------------------------------------------------

        manifest[filename] = {
            "hash": file_hash,
            "chunk_ids": ids,
        }

        self._save_manifest(
            manifest
        )

        new_kb_state = increment_kb_version(self)

        return {
            "filename": filename,
            "chunks_added": len(ids),
            "collection_count": new_kb_state.chroma_chunks,
            "skipped": False,
            "hash": file_hash,
            "chunks_created": len(new_chunks),
            "old_chunks": len(old_ids),
            "stale_chunks_deleted": len(
                stale_old_ids
            ),
            "kb_version": new_kb_state.version,
            "indexes_consistent": new_kb_state.indexes_consistent,
        }

    # ==================================================================
    # KNOWLEDGE BASE STATE & DOCUMENT MANAGEMENT
    # ==================================================================

    def get_kb_state(self) -> dict[str, Any]:
        """Return the current version, document count, and index counts."""
        return get_current_kb_state(self).to_dict()

    def check_index_consistency(self) -> dict[str, Any]:
        """Check whether Chroma and BM25 indexes represent the same chunk count."""
        return check_index_consistency(self)

    def delete_document(
        self,
        filename: str,
        delete_file_from_disk: bool = True,
    ) -> dict[str, Any]:
        """
        Safely remove all chunks of a document from both Chroma and BM25 simultaneously.
        """
        return _delete_kb_document(
            self,
            filename,
            delete_file_from_disk=delete_file_from_disk,
        )

    def _manifest_path(
        self,
    ) -> Path:
        return (
            self.vectorstore_path
            / "ingestion_manifest.json"
        )

    def _load_manifest(
        self,
    ) -> dict[str, Any]:
        path = self._manifest_path()

        if path.exists():
            try:
                return json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                logger.warning(
                    "Could not read ingestion manifest."
                )

        return {}

    def _save_manifest(
        self,
        manifest: dict[str, Any],
    ) -> None:
        path = self._manifest_path()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _build_chunk_metadata(
        chunk: dict[str, Any],
    ) -> dict[str, Any]:
        meta = {
            "filename": chunk.get(
                "filename",
                "",
            ),
            "chunk_id": chunk.get(
                "chunk_id",
                0,
            ),
            "source_path": chunk.get(
                "source_path",
                "",
            ),
            "category": chunk.get(
                "category",
                "",
            )
            or "",
            "section_heading": chunk.get(
                "section_heading",
                "",
            )
            or "",
            "section_path": str(
                chunk.get(
                    "section_path",
                    [],
                )
            ),
            "section_level": chunk.get(
                "section_level",
                0,
            ),
            "chunk_index": chunk.get(
                "chunk_index",
                0,
            ),
            "total_section_chunks": chunk.get(
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
    # MAIN ENTRY POINT
    # ==================================================================

    def _answer_question_impl(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
    ) -> dict[str, Any]:
        """
        Answer a question using the RAG pipeline.

        Flow:
            Validate
              ↓
            Greeting
              ↓
            Query preparation
              ↓
            Query embedding
              ↓
            Dense + BM25
              ↓
            RRF
              ↓
            Evidence gate
              ↓
            Token Optimizer
              ↓
            Context
              ↓
            LLM
              ↓
            Answer
        """

        mode_aliases = {
            "vector": "vector",
            "dense": "vector",
            "bm25": "bm25",
            "hybrid": "hybrid",
            "both": "hybrid",
        }

        retrieval_mode = mode_aliases.get(
            str(retrieval_mode).strip().lower(),
            "hybrid",
        )

        try:
            kb_state = self.get_kb_state()
        except Exception:
            kb_state = {
                "version": 1,
                "document_count": 0,
                "chroma_chunks": 0,
                "bm25_chunks": 0,
                "indexes_consistent": True,
                "last_updated": "",
                "consistency_message": "KB state unavailable",
            }

        def empty_response(
            answer: str,
            response_type: str = "system",
        ) -> dict[str, Any]:
            return {
                "answer": answer,
                "sources": [],
                "retrieved_chunks": [],
                "num_retrieved": 0,
                "evidence": None,
                "evaluation": None,
                "citations": [],
                "citation_coverage": None,
                "confidence": None,
                "kb_state": kb_state,
                "response_type": response_type,
                "retrieval_mode": retrieval_mode,
            }

        # --------------------------------------------------------------
        # 1. VALIDATE QUESTION
        # --------------------------------------------------------------

        if (
            not question
            or not question.strip()
        ):
            return empty_response(
                "Please enter a question.",
                response_type="empty",
            )

        question = question.strip()

        # --------------------------------------------------------------
        # 2. GREETING CHECK
        # --------------------------------------------------------------

        if is_greeting(question):
            return empty_response(
                "Hi! How can I help you?",
                response_type="greeting",
            )

        # --------------------------------------------------------------
        # 3. VALIDATE KNOWLEDGE BASE
        # --------------------------------------------------------------

        if not self.knowledge_base_path.exists():
            return empty_response(
                "The knowledge base does not exist. "
                "Add documents and run ingestion.",
                response_type="error",
            )

        # --------------------------------------------------------------
        # 4. VALIDATE VECTOR STORE
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
                "The vector store could not be initialized.",
                response_type="error",
            )

        if collection_count == 0:
            logger.info(
                "Vector store is empty; "
                "continuing with available lexical retrieval."
            )

        # --------------------------------------------------------------
        # 5. QUERY PREPARATION
        # --------------------------------------------------------------

        search_text = (
            self._prepare_search_query(
                question
            )
        )

        # Capture normalization detail for Observatory (additive)
        try:
            from src.query_normalizer import (
                canonicalize_query as _cq,
            )
            _query_norm_detail = _cq(question)
        except Exception:
            _query_norm_detail = {
                "original": question,
                "cleaned": question,
                "canonical": None,
                "matched_alias": None,
                "expanded_query": search_text,
            }

        logger.info(
            "Original query: %s",
            question,
        )

        logger.info(
            "Search query: %s",
            search_text,
        )

        # --------------------------------------------------------------
        # 6. QUERY EMBEDDING
        # --------------------------------------------------------------

        query_embedding = None

        if (
            collection_count > 0
            and retrieval_mode in {"vector", "hybrid"}
        ):
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
        # 7. DENSE RETRIEVAL
        # --------------------------------------------------------------

        candidates: dict[
            str,
            Candidate,
        ] = {}

        if (
            retrieval_mode in {"vector", "hybrid"}
            and query_embedding
        ):
            candidates = (
                self._dense_retrieve(
                    query_embedding
                )
            )
            for c in candidates.values():
                c.in_dense = True

        # OBSERVATORY: snapshot raw Dense results BEFORE BM25 merges in
        raw_dense_results = [
            c.to_dict()
            for c in sorted(
                candidates.values(),
                key=lambda c: (c.dense_rank or 9999),
            )
            if c.dense_rank is not None
        ]

        # --------------------------------------------------------------
        # 8. BM25 RETRIEVAL
        # --------------------------------------------------------------

        # Capture tokenized query for Observatory display
        bm25_query_tokens: list[str] = []
        if self.bm25 is not None:
            try:
                bm25_query_tokens = self.bm25._tokenize(
                    search_text
                )
            except Exception:
                bm25_query_tokens = []

        if retrieval_mode in {"bm25", "hybrid"}:
            self._bm25_retrieve(
                search_text,
                candidates,
            )
            for c in candidates.values():
                if c.bm25_rank is not None:
                    c.in_bm25 = True

        # OBSERVATORY: snapshot raw BM25 results (ranks now populated)
        raw_bm25_results = [
            c.to_dict()
            for c in sorted(
                candidates.values(),
                key=lambda c: (c.bm25_rank or 9999),
            )
            if c.bm25_rank is not None
        ]

        # --------------------------------------------------------------
        # 9. NO RETRIEVAL RESULTS
        # --------------------------------------------------------------

        if not candidates:
            if self.bm25 is not None:
                return {
                    "answer": ABSTAIN_MESSAGE,
                    "sources": [],
                    "retrieved_chunks": [],
                    "num_retrieved": 0,
                    "evidence": {
                        "level": "none",
                        "should_answer": False,
                        "top_score": 0.0,
                        "score_source": "retrieval",
                        "supporting_chunks": 0,
                        "top_gap": None,
                        "agreement": False,
                        "top_percentile": None,
                        "retrieval_mode": retrieval_mode,
                        "reason": (
                            f"No usable candidates were retrieved "
                            f"in {retrieval_mode} mode."
                        ),
                    },
                    "response_type": "abstain",
                }

            fallback_response = (
                self._answer_via_keyword_fallback(
                    question,
                    search_text,
                )
            )

            fallback_response.setdefault(
                "response_type",
                "fallback",
            )
            fallback_response.setdefault(
                "retrieval_mode",
                retrieval_mode,
            )

            return fallback_response

        # --------------------------------------------------------------
        # 10. RETRIEVAL MODE / FUSION
        # --------------------------------------------------------------

        rrf_results_snapshot: list[dict[str, Any]] = []
        rrf_calculations: list[dict[str, Any]] = []

        if retrieval_mode == "hybrid":
            fused = self._fuse_rrf(
                candidates
            )

            rrf_results_snapshot = [
                c.to_dict()
                for c in fused
            ]

            for candidate in fused:
                dense_contribution = None
                bm25_contribution = None

                if candidate.dense_rank is not None:
                    dense_contribution = 1.0 / (
                        RRF_K + candidate.dense_rank
                    )

                if candidate.bm25_rank is not None:
                    bm25_contribution = 1.0 / (
                        RRF_K + candidate.bm25_rank
                    )

                rrf_calculations.append({
                    "filename": candidate.filename,
                    "chunk_id": candidate.chunk_id,
                    "dense_rank": candidate.dense_rank,
                    "bm25_rank": candidate.bm25_rank,
                    "k": RRF_K,
                    "dense_contribution": dense_contribution,
                    "bm25_contribution": bm25_contribution,
                    "rrf_score": candidate.rrf_score,
                })

        elif retrieval_mode == "vector":
            fused = sorted(
                candidates.values(),
                key=lambda c: (
                    c.dense_rank
                    if c.dense_rank is not None
                    else 9999
                ),
            )

        elif retrieval_mode == "bm25":
            fused = sorted(
                candidates.values(),
                key=lambda c: (
                    c.bm25_rank
                    if c.bm25_rank is not None
                    else 9999
                ),
            )

        else:
            fused = []

        if not fused:
            return {
                "answer": ABSTAIN_MESSAGE,
                "sources": [],
                "retrieved_chunks": [],
                "num_retrieved": 0,
                "evidence": {
                    "level": "none",
                    "should_answer": False,
                    "top_score": 0.0,
                    "score_source": retrieval_mode,
                    "supporting_chunks": 0,
                    "top_gap": None,
                    "agreement": False,
                    "top_percentile": None,
                    "retrieval_mode": retrieval_mode,
                    "reason": (
                        f"No usable candidates remained in "
                        f"{retrieval_mode} mode."
                    ),
                },
                "response_type": "abstain",
                "retrieval_mode": retrieval_mode,
            }

        logger.info(
            "%s candidate pool: %d",
            retrieval_mode,
            len(fused),
        )

        # OBSERVATORY: RRF-ranked snapshot
        for c in fused:
            c.in_rrf = True
        rrf_results_snapshot = [c.to_dict() for c in fused]

        # --------------------------------------------------------------
        # 11. CROSS-ENCODER RERANKING (disabled by design)
        # --------------------------------------------------------------

        reranked = self._rerank(
            question,
            fused,
        )

        # --------------------------------------------------------------
        # 12. EVIDENCE GATE
        # --------------------------------------------------------------

        final_candidates, evidence = (
            self._evaluate_evidence(
                fused,
                reranked,
                search_text,
                retrieval_mode=retrieval_mode,
            )
        )
        for c in final_candidates:
            c.in_final_evidence = True

        # --------------------------------------------------------------
        # RUNTIME EVALUATION LAYER (7 Criteria + Health Score)
        # --------------------------------------------------------------
        eval_result = evaluate_retrieval(
            query=question,
            candidates=list(candidates.values()),
            dense_results=raw_dense_results,
            bm25_results=raw_bm25_results,
            fused_results=fused,
            selected_candidates=final_candidates,
            retrieval_mode=retrieval_mode,
        )

        if not evidence.should_answer:
            from src.rag_pipeline.models import CitationCoverage
            conf_result = evaluate_confidence(
                evaluation=eval_result,
                citations=[],
                citation_coverage=CitationCoverage(0, 0, 0, 0.0, False, []),
                evidence_level=evidence.level,
            )

            return {
                "answer": ABSTAIN_MESSAGE,
                "raw_answer": ABSTAIN_MESSAGE,
                "sources": [],
                "retrieved_chunks": [],
                "num_retrieved": 0,
                "evidence": evidence.as_dict(),
                "evaluation": eval_result.to_dict(),
                "citations": [],
                "citation_coverage": {
                    "total_claims": 0,
                    "supported_claims": 0,
                    "unsupported_claims": 0,
                    "coverage_percentage": 0.0,
                    "has_unsupported": False,
                    "unsupported_claims_list": [],
                },
                "confidence": conf_result.to_dict(),
                "kb_state": kb_state,
                "response_type": "abstain",
                "retrieval_mode": retrieval_mode,
                "trace": {
                    "query": {
                        "original": question,
                        "prepared": search_text,
                        "normalization": _query_norm_detail,
                        "bm25_tokens": bm25_query_tokens,
                        "is_greeting": False,
                        "is_multi_intent": (
                            self._is_multi_intent_query(
                                question
                            )
                        ),
                    },
                    "retrieval": {
                        "mode": retrieval_mode,
                        "selected_mode": retrieval_mode,
                        "candidate_count": len(candidates),
                        "fused_count": len(fused),
                        "reranked": reranked,
                        "dense_available": bool(query_embedding),
                        "bm25_available": (
                            self.bm25 is not None
                        ),
                        "rrf_k": RRF_K,
                        "query_embedding_dim": (
                            len(query_embedding)
                            if query_embedding
                            else 0
                        ),
                        "vector_top_k": VECTOR_TOP_K,
                        "bm25_top_k": BM25_TOP_K,
                        "raw_dense": raw_dense_results,
                        "raw_bm25": raw_bm25_results,
                        "rrf_results": rrf_results_snapshot,
                        "rrf_calculations": rrf_calculations,
                    },
                    "evidence_gate": {
                        "level": evidence.level,
                        "should_answer": False,
                        "top_score": evidence.top_score,
                        "agreement": evidence.agreement,
                        "reason": evidence.reason,
                        "supporting_candidates": [],
                        "rejected_candidates": rrf_results_snapshot,
                    },
                    "evaluation": eval_result.to_dict(),
                    "citations": [],
                    "citation_coverage": {
                        "total_claims": 0,
                        "supported_claims": 0,
                        "unsupported_claims": 0,
                        "coverage_percentage": 0.0,
                        "has_unsupported": False,
                        "unsupported_claims_list": [],
                    },
                    "confidence": conf_result.to_dict(),
                    "kb_state": kb_state,
                    "selected_evidence": [],
                    "candidate_journey": [c.to_dict() for c in fused],
                    "token_optimizer": {},
                    "context": {
                        "chars": 0,
                        "chunks": 0,
                        "sources": [],
                        "final_chunks": [],
                    },
                    "generation": {},
                },
            }

        # OBSERVATORY: evidence gate breakdown
        final_keys = {c.key for c in final_candidates}
        supporting_snapshot = [c.to_dict() for c in final_candidates]
        rejected_snapshot = [
            c.to_dict()
            for c in fused
            if c.key not in final_keys
        ]

        # --------------------------------------------------------------
        # 13. TOKEN OPTIMIZER
        # --------------------------------------------------------------

        from src.rag_pipeline.token_optimizer import (
            optimize_context,
        )

        opt_result = optimize_context(
            final_candidates,
            search_text,
        )

        final_candidates = opt_result["selected"]
        for c in final_candidates:
            c.in_llm_context = True

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
                "answer": ABSTAIN_MESSAGE,
                "raw_answer": ABSTAIN_MESSAGE,
                "sources": [],
                "retrieved_chunks": [],
                "num_retrieved": 0,
                "evidence": (
                    fallback_evidence.as_dict()
                ),
                "evaluation": eval_result.to_dict(),
                "citations": [],
                "citation_coverage": None,
                "confidence": None,
                "kb_state": kb_state,
                "response_type": "abstain",
                "retrieval_mode": retrieval_mode,
            }

        # --------------------------------------------------------------
        # 14. BUILD CONTEXT
        # --------------------------------------------------------------

        (
            context,
            source_names,
            retrieved_chunks,
        ) = self._build_context(
            final_candidates
        )

        # --------------------------------------------------------------
        # 15. GENERATE ANSWER
        # --------------------------------------------------------------

        generation_trace: dict[
            str,
            Any,
        ] = {}

        raw_answer = self._ask_llm(
            question,
            context,
            trace=generation_trace,
        )

        # --------------------------------------------------------------
        # 16. POST-GENERATION GROUNDING & CLAIM CITATIONS
        # --------------------------------------------------------------
        citations, citation_coverage = generate_claim_citations(
            raw_answer,
            final_candidates,
            kb_version=kb_state.get("version", 1),
        )
        answer_evaluation = evaluate_answer_quality(
            question, raw_answer, citations, citation_coverage, final_candidates
        )
        grounding_trace = {
            "stage": "post_generation",
            "claim_count": citation_coverage.total_claims,
            "directly_supported_claims": citation_coverage.supported_claims,
            "partially_supported_claims": citation_coverage.partially_supported_claims,
            "unsupported_claims": citation_coverage.unsupported_claims,
            "coverage_percentage": citation_coverage.coverage_percentage,
            "claim_to_citation": [
                {
                    "claim": citation.claim,
                    "citation": citation.marker,
                    "status": citation.status,
                    "evidence": {
                        "filename": citation.filename,
                        "chunk_id": citation.chunk_id,
                        "passage": citation.passage,
                    },
                }
                for citation in citations
            ],
        }

        # --------------------------------------------------------------
        # 17. EVIDENCE-BACKED CONFIDENCE
        # --------------------------------------------------------------
        confidence_result = evaluate_confidence(
            evaluation=eval_result,
            citations=citations,
            citation_coverage=citation_coverage,
            evidence_level=evidence.level,
        )

        answer = (
            f"{raw_answer}\n\n"
            f"Evidence-backed Confidence: {confidence_result.level} "
            f"({evidence.level.title()} evidence)"
        )

        opt_trace: dict[str, Any] = {
            **opt_result["stats"],
            "steps": opt_result["steps"],
            "removed_chunks": opt_result["removed"],
        }

        return {
            "answer": answer,
            "raw_answer": raw_answer,
            "sources": source_names,
            "retrieved_chunks": retrieved_chunks,
            "num_retrieved": len(
                retrieved_chunks
            ),
            "evidence": evidence.as_dict(),
            "evaluation": eval_result.to_dict(),
            "citations": [c.to_dict() for c in citations],
            "citation_coverage": citation_coverage.to_dict(),
            "grounding": grounding_trace,
            "confidence": confidence_result.to_dict(),
            "answer_evaluation": answer_evaluation,
            "kb_state": kb_state,
            "response_type": "rag",
            "retrieval_mode": retrieval_mode,
            "trace": {
                "query": {
                    "original": question,
                    "prepared": search_text,
                    "normalization": _query_norm_detail,
                    "bm25_tokens": bm25_query_tokens,
                    "is_greeting": False,
                    "is_multi_intent": (
                        self._is_multi_intent_query(
                            question
                        )
                    ),
                },
                "retrieval": {
                    "mode": retrieval_mode,
                    "selected_mode": retrieval_mode,
                    "candidate_count": len(
                        candidates
                    ),
                    "fused_count": len(
                        fused
                    ),
                    "reranked": reranked,
                    "dense_available": (
                        bool(query_embedding)
                    ),
                    "bm25_available": (
                        self.bm25 is not None
                    ),
                    "rrf_k": RRF_K,
                    "query_embedding_dim": (
                        len(query_embedding)
                        if query_embedding
                        else 0
                    ),
                    "vector_top_k": VECTOR_TOP_K,
                    "bm25_top_k": BM25_TOP_K,
                    "raw_dense": raw_dense_results,
                    "raw_bm25": raw_bm25_results,
                    "rrf_results": rrf_results_snapshot,
                    "rrf_calculations": rrf_calculations,
                },
                "evidence_gate": {
                    "level": evidence.level,
                    "should_answer": evidence.should_answer,
                    "top_score": evidence.top_score,
                    "agreement": evidence.agreement,
                    "reason": evidence.reason,
                    "supporting_candidates": supporting_snapshot,
                    "rejected_candidates": rejected_snapshot,
                },
                "evaluation": eval_result.to_dict(),
                "citations": [c.to_dict() for c in citations],
                "citation_coverage": citation_coverage.to_dict(),
                "grounding": grounding_trace,
                "confidence": confidence_result.to_dict(),
                "answer_evaluation": answer_evaluation,
                "kb_state": kb_state,
                "selected_evidence": [c.to_dict() for c in final_candidates],
                "candidate_journey": [c.to_dict() for c in fused],
                "token_optimizer": opt_trace,
                "context": {
                    "chars": len(context),
                    "chunks": len(
                        retrieved_chunks
                    ),
                    "sources": source_names,
                    "final_chunks": retrieved_chunks,
                },
                "generation": (
                    generation_trace
                ),
            },
        }
