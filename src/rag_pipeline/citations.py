from __future__ import annotations

import re
from pathlib import Path

from src.rag_pipeline.models import Candidate, Citation, CitationCoverage

_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with", "which", "who", "what", "when", "where", "why", "how", "has", "have"}


def _split_into_claims(answer: str) -> list[str]:
    """Extract answer statements, excluding presentation-only sections."""
    lines = []
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(?:sources|confidence|references|note|details|answer):\b", line, re.I):
            break
        lines.append(re.sub(r"^(?:[-*]|\d+\.)\s+", "", line))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(lines)) if len(s.strip()) >= 15]


def _find_best_passage_in_chunk(claim: str, text: str) -> tuple[str, float]:
    claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower())) - _STOPWORDS
    if not claim_terms or not text.strip():
        return (text[:250].strip() or ""), 0.0

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    best_sent, best_score = "", 0.0
    for idx, sentence in enumerate(sentences):
        terms = set(re.findall(r"[a-z0-9]+", sentence.lower())) - _STOPWORDS
        score = len(claim_terms & terms) / max(1, len(claim_terms))
        if score > best_score:
            # Expand to include adjacent sentence if helpful for context
            ctx_start = max(0, idx - 1)
            ctx_end = min(len(sentences), idx + 2)
            best_sent = " ".join(sentences[ctx_start:ctx_end])
            best_score = score

    return (best_sent or text[:300].strip()), best_score


def _claim_support(claim: str, passage: str, chunk_text: str = "") -> tuple[str, float, str]:
    """Verify direct support against passage and full chunk context."""
    claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower())) - _STOPWORDS
    search_scope = f"{passage} {chunk_text}".lower()
    passage_terms = set(re.findall(r"[a-z0-9]+", search_scope)) - _STOPWORDS

    if not claim_terms:
        return "supported", 1.0, "Claim contains no specific content constraints."

    coverage = len(claim_terms & passage_terms) / max(1, len(claim_terms))

    # Verify key numeric values if present in the claim
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    passage_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", search_scope))
    missing_values = claim_numbers - passage_numbers
    if missing_values:
        detail = ", ".join(sorted(missing_values))
        return "unsupported", coverage, f"Related text, but key numeric value is absent: {detail}"

    if coverage >= 0.45:
        return "supported", coverage, "Direct answer-bearing terms are present in the passage."
    if coverage >= 0.25:
        return "partially_supported", coverage, "The passage provides partial support for the claim."
    return "unsupported", coverage, "The passage lacks sufficient direct evidence for this claim."


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
        status, support_score, detail = _claim_support(display_claim, passage, chunk_text=candidate.text or "")
        if status == "supported":
            fully_supported += 1
        elif status == "partially_supported":
            partial += 1
        else:
            unsupported.append(display_claim)

        meta = candidate.metadata if candidate else {}
        file_type = meta.get("file_type") or (Path(candidate.filename).suffix.lstrip(".").lower() if candidate and candidate.filename else "")
        page_num = meta.get("page_number")
        try:
            page_number = int(page_num) if page_num and int(page_num) > 0 else None
        except (ValueError, TypeError):
            page_number = None

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
            file_type=file_type,
            page_number=page_number,
        ))
    total = len(claims)
    cov_pct = round(((fully_supported * 1.0 + partial * 0.5) / total * 100.0) if total else 100.0, 1)
    return citations, CitationCoverage(
        total_claims=total, supported_claims=fully_supported, unsupported_claims=len(unsupported),
        coverage_percentage=cov_pct,
        has_unsupported=bool(unsupported), unsupported_claims_list=unsupported,
        partially_supported_claims=partial,
    )

