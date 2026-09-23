from __future__ import annotations

import re

from src.rag_pipeline.models import Candidate, Citation, CitationCoverage

_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with", "which", "who", "what", "when", "where", "why", "how", "has", "have"}


def _split_into_claims(answer: str) -> list[str]:
    """Extract answer statements, excluding presentation-only sections."""
    lines = []
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(?:sources|confidence|references|note):\b", line, re.I):
            break
        lines.append(re.sub(r"^(?:[-*]|\d+\.)\s+", "", line))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(lines)) if len(s.strip()) >= 15]


def _find_best_passage_in_chunk(claim: str, text: str) -> tuple[str, float]:
    claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower())) - _STOPWORDS
    best, best_score = "", 0.0
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        terms = set(re.findall(r"[a-z0-9]+", sentence.lower())) - _STOPWORDS
        score = len(claim_terms & terms) / max(1, len(claim_terms))
        if score > best_score:
            best, best_score = sentence.strip(), score
    return (best or text[:250].strip()), best_score


def _claim_support(claim: str, passage: str) -> tuple[str, float, str]:
    """Verify direct support; reject related lexical matches with wrong facts."""
    claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower())) - _STOPWORDS
    passage_terms = set(re.findall(r"[a-z0-9]+", passage.lower())) - _STOPWORDS
    coverage = len(claim_terms & passage_terms) / max(1, len(claim_terms))
    entities = re.findall(r"(?<!^)\b[A-Z][A-Za-z0-9-]+\b", claim)
    missing_entities = [term for term in entities if term.lower() not in passage_terms]
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    passage_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", passage))
    missing_values = claim_numbers - passage_numbers
    if missing_entities or missing_values:
        detail = ", ".join(missing_entities + sorted(missing_values))
        return "unsupported", coverage, "Related lexical match, but answer-bearing entity/value is absent: " + detail
    if coverage >= 0.70:
        return "supported", coverage, "Direct answer-bearing terms are present in the passage."
    if coverage >= 0.40:
        return "partially_supported", coverage, "The passage supports only part of the claim."
    return "unsupported", coverage, "The passage is topically related but lacks direct answer evidence."


def _origin(candidate: Candidate) -> str:
    parts = []
    if candidate.dense_rank is not None:
        parts.append(f"Dense #{candidate.dense_rank}")
    if candidate.bm25_rank is not None:
        parts.append(f"BM25 #{candidate.bm25_rank}")
    if candidate.rrf_score:
        parts.append(f"RRF {candidate.rrf_score:.4f}")
    return ", ".join(parts) or "Retrieved context"


def generate_claim_citations(answer: str, selected_candidates: list[Candidate], kb_version: int | str = 1) -> tuple[list[Citation], CitationCoverage]:
    """Post-generation grounding: Claim -> exact evidence passage -> citation verdict."""
    claims = _split_into_claims(answer) if answer.strip() else []
    if not selected_candidates:
        return [], CitationCoverage(len(claims), 0, len(claims), 0.0, bool(claims), claims)
    citations: list[Citation] = []
    unsupported: list[str] = []
    fully_supported = partial = 0
    chunk_map = {index + 1: candidate for index, candidate in enumerate(selected_candidates)}
    for claim_id, claim in enumerate(claims, start=1):
        display_claim = re.sub(r"\[(?:Source\s*)?\d+\]", "", claim).strip()
        explicit = [int(v) for v in re.findall(r"\[(?:Source\s*)?(\d+)\]", claim) if int(v) in chunk_map]
        candidates = [chunk_map[index] for index in explicit] if explicit else selected_candidates
        scored = [(_find_best_passage_in_chunk(display_claim, candidate.text or ""), candidate) for candidate in candidates]
        (passage, lexical), candidate = max(scored, key=lambda item: item[0][1])
        status, support_score, detail = _claim_support(display_claim, passage)
        if status == "supported":
            fully_supported += 1
        elif status == "partially_supported":
            partial += 1
        else:
            unsupported.append(display_claim)
        citations.append(Citation(
            id=claim_id, marker=f"[{claim_id}]", claim=display_claim,
            filename=candidate.filename if status != "unsupported" or explicit else "None",
            chunk_id=candidate.chunk_id if status != "unsupported" or explicit else "N/A",
            passage=passage if status != "unsupported" or explicit else "[No supporting passage found in retrieved evidence]",
            retrieval_source=_origin(candidate) if status != "unsupported" or explicit else "Unverified",
            status=status,
            verification_reason=f"{status.replace('_', ' ').title()}: {detail} (term coverage: {support_score:.0%})",
            kb_version=kb_version, support_score=support_score, lexical_overlap=lexical,
            direct_answer_support=status == "supported",
        ))
    total = len(claims)
    return citations, CitationCoverage(
        total_claims=total, supported_claims=fully_supported, unsupported_claims=len(unsupported),
        coverage_percentage=(fully_supported / total * 100.0) if total else 100.0,
        has_unsupported=bool(unsupported), unsupported_claims_list=unsupported,
        partially_supported_claims=partial,
    )
