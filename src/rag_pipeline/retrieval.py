from __future__ import annotations

import logging
from typing import Any

from src.config import BM25_TOP_K, VECTOR_TOP_K
from src.rag_pipeline.config import (
    ENABLE_RERANKER,
    MAX_RERANK_CANDIDATES,
    RERANK_CANDIDATE_LIMIT,
    RRF_K,
)
from src.rag_pipeline.models import Candidate
from src.rag_pipeline.query import candidate_key, safe_float

logger = logging.getLogger(__name__)


# ======================================================================
# DENSE / VECTOR RETRIEVAL
# ======================================================================


def dense_retrieve(
    vector_store: Any,
    query_embedding: list[float],
    top_k: int = VECTOR_TOP_K,
) -> dict[str, Candidate]:
    """
    Retrieve chunks from the persistent Chroma vector store.

    Returns:
        dict[str, Candidate]
        Candidate key is filename + chunk_id.
    """

    retrieval_k = min(
        max(
            VECTOR_TOP_K,
            top_k,
            RERANK_CANDIDATE_LIMIT,
        ),
        MAX_RERANK_CANDIDATES,
    )

    if not query_embedding:
        return {}

    try:
        results = vector_store.query(
            query_embedding,
            n_results=retrieval_k,
        )
    except Exception:
        logger.exception("Dense retrieval failed.")
        return {}

    if not isinstance(results, dict):
        logger.warning(
            "Dense retrieval returned unexpected type: %s",
            type(results).__name__,
        )
        return {}

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") or []

    # Chroma normally returns:
    # [[doc1, doc2, ...]]
    #
    # Normalize both nested and flat forms.
    if documents and isinstance(documents[0], list):
        documents = documents[0]

    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]

    if distances and isinstance(distances[0], list):
        distances = distances[0]

    candidates: dict[str, Candidate] = {}

    for index, document in enumerate(documents):
        if document is None:
            continue

        text = str(document).strip()

        if not text:
            continue

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            and isinstance(metadatas[index], dict)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        filename = str(
            metadata.get(
                "filename",
                f"document-{index}",
            )
        )

        chunk_id = metadata.get(
            "chunk_id",
            index,
        )

        key = candidate_key(
            filename,
            chunk_id,
        )

        candidates[key] = Candidate(
            key=key,
            filename=filename,
            chunk_id=chunk_id,
            text=text,
            metadata=dict(metadata),
            dense_rank=index + 1,
            dense_distance=safe_float(distance),
        )

    logger.debug(
        "Dense retrieval returned %d candidates.",
        len(candidates),
    )

    return candidates


# ======================================================================
# BM25 RETRIEVAL
# ======================================================================


def bm25_retrieve(
    bm25: Any,
    search_text: str,
    candidates: dict[str, Candidate],
) -> None:
    """
    Retrieve lexical matches using the project's existing
    src.bm25_retriever.BM25Retriever.

    IMPORTANT:
        The existing BM25 implementation exposes:

            bm25.query(query, top_k)

        We use that API directly.

    Dense and BM25 candidates are merged using:

        filename + chunk_id

    so that Hybrid mode can calculate RRF correctly.
    """

    if bm25 is None:
        logger.warning(
            "BM25 retrieval requested but BM25 retriever is None."
        )
        return

    search_text = (search_text or "").strip()

    if not search_text:
        return

    retrieval_k = min(
        max(
            BM25_TOP_K,
            RERANK_CANDIDATE_LIMIT,
        ),
        MAX_RERANK_CANDIDATES,
    )

    try:
        # This is the actual API from src/bm25_retriever.py.
        bm25_results = bm25.query(
            search_text,
            top_k=retrieval_k,
        )
    except Exception:
        logger.exception(
            "BM25 retrieval failed for query: %r",
            search_text,
        )
        return

    if not bm25_results:
        logger.debug(
            "BM25 returned no results for query: %r",
            search_text,
        )
        return

    for index, result in enumerate(bm25_results):

        if not isinstance(result, dict):
            continue

        rank = index + 1

        filename = str(
            result.get(
                "filename",
                "unknown",
            )
        )

        chunk_id = result.get(
            "chunk_id",
            index,
        )

        text = str(
            result.get(
                "text",
                "",
            )
            or ""
        ).strip()

        if not text:
            continue

        key = candidate_key(
            filename,
            chunk_id,
        )

        # --------------------------------------------------------------
        # Merge with an existing Dense candidate
        # --------------------------------------------------------------

        if key not in candidates:

            metadata = {
                "filename": filename,
                "chunk_id": chunk_id,
            }

            for field_name in (
                "source_path",
                "section_heading",
                "section_path",
                "section_level",
                "chunk_index",
                "total_section_chunks",
                "category",
                "has_heading",
            ):
                value = result.get(field_name)

                if value is not None:
                    metadata[field_name] = value

            candidates[key] = Candidate(
                key=key,
                filename=filename,
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
            )

        candidate = candidates[key]

        # --------------------------------------------------------------
        # BM25 ranking information
        # --------------------------------------------------------------

        candidate.bm25_rank = rank

        candidate.bm25_score = (
            safe_float(
                result.get(
                    "bm25_score",
                    0.0,
                ),
                default=0.0,
            )
            or 0.0
        )

        # --------------------------------------------------------------
        # Preserve all useful BM25 metadata
        # --------------------------------------------------------------

        for field_name in (
            "source_path",
            "section_heading",
            "section_path",
            "section_level",
            "chunk_index",
            "total_section_chunks",
            "category",
            "has_heading",
        ):
            value = result.get(field_name)

            if value is not None and value != "":
                candidate.metadata[field_name] = value

    logger.debug(
        "BM25 retrieval added/merged candidates; total pool=%d.",
        len(candidates),
    )


# ======================================================================
# RECIPROCAL RANK FUSION
# ======================================================================


def calculate_rrf_score(
    candidate: Candidate,
    rrf_k: int = RRF_K,
) -> float:
    """
    Calculate the actual RRF score.

    RRF is based on rank, NOT raw Dense distance or BM25 score.

        RRF =
            1 / (K + DenseRank)
            +
            1 / (K + BM25Rank)
    """

    score = 0.0

    if candidate.dense_rank is not None:
        score += 1.0 / (
            rrf_k + candidate.dense_rank
        )

    if candidate.bm25_rank is not None:
        score += 1.0 / (
            rrf_k + candidate.bm25_rank
        )

    return score


def get_rrf_contributions(
    candidate: Candidate,
    rrf_k: int = RRF_K,
) -> dict[str, Any]:
    """
    Return the individual RRF contributions for Observatory /
    Retrieval Lab diagnostics.
    """

    dense_contribution = None
    bm25_contribution = None

    if candidate.dense_rank is not None:
        dense_contribution = 1.0 / (
            rrf_k + candidate.dense_rank
        )

    if candidate.bm25_rank is not None:
        bm25_contribution = 1.0 / (
            rrf_k + candidate.bm25_rank
        )

    score = (
        (dense_contribution or 0.0)
        + (bm25_contribution or 0.0)
    )

    return {
        "filename": candidate.filename,
        "chunk_id": candidate.chunk_id,
        "dense_rank": candidate.dense_rank,
        "bm25_rank": candidate.bm25_rank,
        "k": rrf_k,
        "dense_contribution": dense_contribution,
        "bm25_contribution": bm25_contribution,
        "rrf_score": score,
    }


def fuse_rrf(
    candidates: dict[str, Candidate],
    rrf_k: int = RRF_K,
    limit: int = RERANK_CANDIDATE_LIMIT,
) -> list[Candidate]:
    """
    Fuse Dense + BM25 candidates using Reciprocal Rank Fusion.

    This function is ONLY used for Hybrid mode.
    """

    if not candidates:
        return []

    for candidate in candidates.values():
        candidate.rrf_score = calculate_rrf_score(
            candidate,
            rrf_k=rrf_k,
        )

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.rrf_score,
            candidate.in_both_retrievers,
        ),
        reverse=True,
    )

    return ranked[:limit]


# ======================================================================
# RETRIEVAL SNAPSHOTS / DIAGNOSTICS
# ======================================================================


def candidate_to_retrieval_dict(
    candidate: Candidate,
) -> dict[str, Any]:
    """
    Convert a Candidate into a safe diagnostic representation.
    """

    return {
        "key": candidate.key,
        "filename": candidate.filename,
        "chunk_id": candidate.chunk_id,
        "dense_rank": candidate.dense_rank,
        "dense_distance": candidate.dense_distance,
        "bm25_rank": candidate.bm25_rank,
        "bm25_score": candidate.bm25_score,
        "rrf_score": candidate.rrf_score,
        "in_both_retrievers": candidate.in_both_retrievers,
        "text": candidate.text,
        "metadata": candidate.metadata,
    }


def get_rrf_results(
    candidates: list[Candidate],
) -> list[dict[str, Any]]:
    """
    Return RRF-ranked candidates for the Retrieval Lab.
    """

    return [
        candidate_to_retrieval_dict(candidate)
        for candidate in candidates
    ]


def get_retrieval_statistics(
    candidates: dict[str, Candidate],
) -> dict[str, Any]:
    """
    Runtime retrieval statistics.
    """

    dense_count = sum(
        1
        for candidate in candidates.values()
        if candidate.dense_rank is not None
    )

    bm25_count = sum(
        1
        for candidate in candidates.values()
        if candidate.bm25_rank is not None
    )

    both_count = sum(
        1
        for candidate in candidates.values()
        if (
            candidate.dense_rank is not None
            and candidate.bm25_rank is not None
        )
    )

    return {
        "candidate_count": len(candidates),
        "dense_count": dense_count,
        "bm25_count": bm25_count,
        "both_count": both_count,
    }


# ======================================================================
# CROSS-ENCODER RERANKING
# ======================================================================


def rerank(
    question: str,
    candidates: list[Candidate],
    reranker: Any = None,
) -> bool:
    """
    CrossEncoder reranking is intentionally disabled.

    The current architecture is:

        Dense
          +
        BM25
          +
        RRF

    No Hugging Face model is downloaded or loaded.
    """

    if not ENABLE_RERANKER:
        return False

    # Keep disabled even if an object is accidentally supplied.
    return False