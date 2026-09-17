
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ======================================================================
# RETRIEVAL CANDIDATE
# ======================================================================


@dataclass
class Candidate:
    """
    Represents one retrieved chunk throughout the retrieval pipeline.

    A Candidate is the common data structure shared by Dense retrieval,
    BM25 retrieval, RRF fusion, evidence evaluation, and context
    selection.

    Flow:

        Dense Retrieval
              +
        BM25 Retrieval
              ↓
          Candidate Pool
              ↓
          RRF Fusion
              ↓
        Evidence Evaluation
              ↓
        Context Selection
              ↓
        LLM Generation
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    key: str
    filename: str
    chunk_id: Any
    text: str

    # Original chunk metadata.
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    dense_rank: int | None = None
    dense_distance: float | None = None

    # ------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------

    bm25_rank: int | None = None
    bm25_score: float | None = None

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    rrf_score: float = 0.0

    # ------------------------------------------------------------------
    # Optional reranking
    # ------------------------------------------------------------------
    #
    # CrossEncoder reranking is currently disabled in this project.
    # The field remains for backward compatibility and for preserving
    # the pipeline's data model.
    #

    rerank_score: float | None = None

    # ------------------------------------------------------------------
    # Retrieval agreement
    # ------------------------------------------------------------------

    @property
    def in_both_retrievers(self) -> bool:
        """
        Return True when the same chunk was retrieved by both
        Dense retrieval and BM25.
        """
        return (
            self.dense_rank is not None
            and self.bm25_rank is not None
        )

    # ------------------------------------------------------------------
    # Observatory representation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Candidate into a serializable dictionary.

        This is useful for the RAG Observatory because the UI can
        inspect the complete state of a candidate without duplicating
        conversion logic.
        """
        return {
            "key": self.key,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": dict(self.metadata),

            # Dense retrieval
            "dense_rank": self.dense_rank,
            "dense_distance": self.dense_distance,

            # BM25 retrieval
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,

            # RRF
            "rrf_score": self.rrf_score,

            # Optional reranker
            "rerank_score": self.rerank_score,

            # Agreement
            "retrieval_agreement": self.in_both_retrievers,
        }


# ======================================================================
# EVIDENCE
# ======================================================================


@dataclass
class Evidence:
    """
    Represents the strength of the retrieved evidence.

    The evidence gate combines:

    - absolute RRF floor
    - RRF rank strength
    - retriever agreement
    - source diversity
    - supporting chunks
    - retrieval mode
    """

    # Final evidence classification.
    #
    # Expected values:
    #   strong
    #   moderate
    #   weak
    #   insufficient / abstain
    level: str

    # Whether the pipeline is allowed to generate an answer.
    should_answer: bool

    # Highest relevant retrieval score.
    top_score: float

    # Where the score originated.
    score_source: str

    # Number of chunks supporting the final decision.
    supporting_chunks: int

    # Difference between top and next-best candidate, when available.
    top_gap: float | None

    # Whether the top evidence appears in both retrievers.
    agreement: bool

    # Relative position/strength of the top candidate.
    top_percentile: float | None

    # Actual retrieval mode used.
    retrieval_mode: str

    # Human-readable explanation of the evidence decision.
    reason: str

    # ------------------------------------------------------------------
    # Dictionary representation
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """
        Convert evidence information into a UI/API-friendly dictionary.
        """
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
