from __future__ import annotations

import math
from typing import Any

from src.rag_pipeline.config import (
    MAX_CHUNKS_PER_SOURCE,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
)
from src.rag_pipeline.models import Candidate
from src.rag_pipeline.query import get_retrieval_mode, normalize_text


def deduplicate_candidates(
    candidates: list[Candidate],
) -> list[Candidate]:
    """
    Remove duplicate candidates by stable key and normalized text.

    Empty candidates are ignored.
    """
    seen_keys: set[str] = set()
    seen_text: set[str] = set()

    unique: list[Candidate] = []

    for candidate in candidates:
        if not candidate.text or not candidate.text.strip():
            continue

        if candidate.key in seen_keys:
            continue

        normalized_text = normalize_text(candidate.text)

        if not normalized_text:
            continue

        if normalized_text in seen_text:
            continue

        seen_keys.add(candidate.key)
        seen_text.add(normalized_text)

        unique.append(candidate)

    return unique


def _safe_rank(
    value: int | None,
) -> int:
    """
    Missing retrieval ranks are placed after ranked candidates.
    """
    if value is None:
        return 999999

    try:
        return int(value)
    except (TypeError, ValueError):
        return 999999


def _safe_score(
    value: float | None,
    default: float = float("-inf"),
) -> float:
    """
    Safely convert a score to float.
    """
    if value is None:
        return default

    try:
        score = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(score):
        return default

    return score


def _candidate_sort_key(
    candidate: Candidate,
    retrieval_mode: str,
    reranked: bool,
) -> tuple[Any, ...]:
    """
    Return a ranking key appropriate for the active retrieval mode.

    Hybrid:
        RRF score

    Vector:
        Dense rank first, then dense distance

    BM25:
        BM25 rank first, then BM25 score

    Reranking:
        Reranker score takes priority when available.
    """

    if reranked:
        return (
            _safe_score(
                candidate.rerank_score,
            ),
        )

    if retrieval_mode == "vector":
        # Dense rank is the primary ordering signal.
        #
        # Lower distance is better, so negate it because the
        # final sort is descending.
        distance = _safe_score(
            candidate.dense_distance,
            default=float("inf"),
        )

        distance_key = (
            -distance
            if math.isfinite(distance)
            else float("-inf")
        )

        return (
            -_safe_rank(candidate.dense_rank),
            distance_key,
        )

    if retrieval_mode == "bm25":
        # BM25 rank is the primary ordering signal.
        # Higher BM25 score is used as a secondary signal.
        return (
            -_safe_rank(candidate.bm25_rank),
            _safe_score(candidate.bm25_score),
        )

    # Hybrid / default.
    return (
        _safe_score(candidate.rrf_score),
    )


def select_context_candidates(
    candidates: list[Candidate],
    reranked: bool,
    diversify_sources: bool = False,
    retrieval_mode: str | None = None,
) -> list[Candidate]:
    """
    Select the candidates that will be passed into the context builder.

    The ranking strategy depends on the actual retrieval mode:

        vector  -> dense rank / distance
        bm25    -> BM25 rank / score
        hybrid  -> RRF score

    This prevents Vector/BM25-only retrieval from incorrectly
    depending on an RRF score that does not exist in those modes.
    """

    if not candidates:
        return []

    # --------------------------------------------------------------
    # DETERMINE RETRIEVAL MODE
    # --------------------------------------------------------------

    mode = (
        str(retrieval_mode).strip().lower()
        if retrieval_mode
        else get_retrieval_mode(candidates)
    )

    mode_aliases = {
        "dense": "vector",
        "vector": "vector",
        "bm25": "bm25",
        "hybrid": "hybrid",
        "both": "hybrid",
    }

    mode = mode_aliases.get(
        mode,
        "hybrid",
    )

    # --------------------------------------------------------------
    # SORT
    # --------------------------------------------------------------

    ranked = sorted(
        candidates,
        key=lambda candidate: _candidate_sort_key(
            candidate,
            mode,
            reranked,
        ),
        reverse=True,
    )

    # --------------------------------------------------------------
    # DEDUPLICATE BEFORE LIMITING
    # --------------------------------------------------------------
    #
    # This allows a lower-ranked candidate to replace a duplicate
    # candidate instead of losing that slot completely.
    # --------------------------------------------------------------

    ranked = deduplicate_candidates(ranked)

    if not ranked:
        return []

    # --------------------------------------------------------------
    # SOURCE DIVERSIFICATION
    # --------------------------------------------------------------

    if diversify_sources:
        selected: list[Candidate] = []
        source_counts: dict[str, int] = {}

        # First pass:
        # Try to give different sources representation.
        for candidate in ranked:
            source = (
                candidate.filename
                or candidate.metadata.get(
                    "filename",
                    "unknown",
                )
                or "unknown"
            )

            count = source_counts.get(
                source,
                0,
            )

            if count >= MAX_CHUNKS_PER_SOURCE:
                continue

            selected.append(candidate)
            source_counts[source] = count + 1

            if len(selected) >= MAX_CONTEXT_CHUNKS:
                break

        ranked = selected

    else:
        # ----------------------------------------------------------
        # NORMAL SOURCE LIMITING
        # ----------------------------------------------------------

        selected = []
        source_counts = {}

        for candidate in ranked:
            source = (
                candidate.filename
                or candidate.metadata.get(
                    "filename",
                    "unknown",
                )
                or "unknown"
            )

            count = source_counts.get(
                source,
                0,
            )

            if count >= MAX_CHUNKS_PER_SOURCE:
                continue

            selected.append(candidate)
            source_counts[source] = count + 1

            if len(selected) >= MAX_CONTEXT_CHUNKS:
                break

        ranked = selected

    # --------------------------------------------------------------
    # CONTEXT CHARACTER LIMIT
    # --------------------------------------------------------------

    final_candidates: list[Candidate] = []
    total_chars = 0

    for candidate in ranked:
        text = (
            candidate.text
            or ""
        ).strip()

        if not text:
            continue

        separator_chars = 2 if final_candidates else 0
        projected_chars = (
            total_chars
            + separator_chars
            + len(text)
        )

        if projected_chars > MAX_CONTEXT_CHARS:
            # If the context is already non-empty, stop here.
            if final_candidates:
                break

            # If even the first chunk is too large, keep a
            # truncated version rather than returning no context.
            available = max(
                0,
                MAX_CONTEXT_CHARS,
            )

            if available <= 0:
                break

            candidate.text = text[:available]
            final_candidates.append(candidate)
            break

        final_candidates.append(candidate)
        total_chars = projected_chars

        if len(final_candidates) >= MAX_CONTEXT_CHUNKS:
            break

    return final_candidates


def build_context(
    candidates: list[Candidate],
) -> tuple[
    str,
    list[str],
    list[dict[str, Any]],
]:
    """
    Build the final LLM context and observability metadata.

    Returns:
        context:
            Text passed to the LLM.

        source_names:
            Unique source filenames.

        retrieved_chunks:
            Structured information about each selected chunk.
    """

    if not candidates:
        return (
            "",
            [],
            [],
        )

    context_parts: list[str] = []
    source_names: list[str] = []
    seen_sources: set[str] = set()
    retrieved_chunks: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        text = (
            candidate.text
            or ""
        ).strip()

        if not text:
            continue

        filename = (
            candidate.filename
            or candidate.metadata.get(
                "filename",
                "unknown",
            )
            or "unknown"
        )

        metadata = candidate.metadata or {}

        if filename not in seen_sources:
            seen_sources.add(filename)
            source_names.append(filename)

        context_parts.append(
            f"[Source {index}: {filename}]\n"
            f"{text}"
        )

        retrieved_chunks.append(
            {
                "rank": index,
                "filename": filename,
                "chunk_id": candidate.chunk_id,
                "text": text,
                "metadata": metadata,
                "dense_rank": candidate.dense_rank,
                "dense_distance": candidate.dense_distance,
                "bm25_rank": candidate.bm25_rank,
                "bm25_score": candidate.bm25_score,
                "rrf_score": candidate.rrf_score,
                "rerank_score": candidate.rerank_score,
                "in_both_retrievers": (
                    candidate.in_both_retrievers
                ),
            }
        )

    context = "\n\n".join(
        context_parts
    )

    return (
        context,
        source_names,
        retrieved_chunks,
    )