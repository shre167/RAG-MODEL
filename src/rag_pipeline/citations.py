from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.rag_pipeline.models import Candidate, Citation, CitationCoverage

if TYPE_CHECKING:
    from src.embeddings import EmbeddingService

_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with", "which", "who", "what", "when", "where", "why", "how", "has", "have", "had", "been", "will", "would", "could", "should", "may", "might", "can"}


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


def _split_into_claims(answer: str) -> list[str]:
    """
    Extract claims ONLY from the Answer section, ignoring Details, Sources, etc.
    
    Expected format:
        Answer:
        <claims to extract>
        
        Details:
        <ignore this>
        
        Sources:
        <ignore this>
    """
    lines = []
    in_answer_section = False
    
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        
        # Check if we're entering the Answer section
        if re.match(r"^(?:\*\*)?answer(?:\*\*)?:\s*$", line, re.I):
            in_answer_section = True
            continue
        
        # Check if we're leaving the Answer section (entering Details, Sources, etc.)
        if re.match(r"^(?:\*\*)?(?:details|sources|confidence|references|note)(?:\*\*)?:\s*$", line, re.I):
            in_answer_section = False
            break  # Stop processing once we leave Answer section
        
        # If we're in the answer section, collect the line
        if in_answer_section:
            # Remove bullet points or numbering
            cleaned = re.sub(r"^(?:[-*]|\d+\.)\s+", "", line)
            lines.append(cleaned)
    
    # If no explicit "Answer:" section found, process all lines until we hit a section marker
    # This handles cases where LLM doesn't use the exact format
    if not lines:
        for line in answer.splitlines():
            line = line.strip()
            if not line:
                continue
            # Stop at any section marker
            if re.match(r"^(?:\*\*)?(?:answer|details|sources|confidence|references|note)(?:\*\*)?:\s*$", line, re.I):
                break
            # Remove bullet points or numbering
            cleaned = re.sub(r"^(?:[-*]|\d+\.)\s+", "", line)
            lines.append(cleaned)
    
    # Split into sentences and filter by minimum length
    combined_text = " ".join(lines)
    claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", combined_text) if len(s.strip()) >= 15]
    
    return claims


def _find_best_passage_semantic(
    claim: str, 
    text: str, 
    embedding_service: "EmbeddingService | None" = None
) -> tuple[str, float]:
    """Find most relevant passage using semantic similarity with embeddings."""
    if not text.strip():
        return "", 0.0
    
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return text[:300].strip(), 0.0
    
    # If no embedding service, fall back to lexical matching
    if embedding_service is None:
        return _find_best_passage_lexical(claim, text)
    
    try:
        # Get claim embedding
        claim_emb = embedding_service.create_embedding(claim)
        
        # Score each sentence semantically
        best_passage = ""
        best_score = 0.0
        
        for idx, sentence in enumerate(sentences):
            sent_emb = embedding_service.create_embedding(sentence)
            similarity = _cosine_similarity(claim_emb, sent_emb)
            
            if similarity > best_score:
                # Include context: sentence before and after
                ctx_start = max(0, idx - 1)
                ctx_end = min(len(sentences), idx + 2)
                best_passage = " ".join(sentences[ctx_start:ctx_end])
                best_score = similarity
        
        return best_passage or text[:300].strip(), best_score
    
    except Exception:
        # Fallback to lexical if embeddings fail
        return _find_best_passage_lexical(claim, text)


def _find_best_passage_lexical(claim: str, text: str) -> tuple[str, float]:
    """Lexical fallback: find passage with most term overlap."""
    claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower())) - _STOPWORDS
    if not claim_terms:
        return text[:300].strip(), 0.0
    
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    best_passage = ""
    best_score = 0.0
    
    for idx, sentence in enumerate(sentences):
        terms = set(re.findall(r"[a-z0-9]+", sentence.lower())) - _STOPWORDS
        overlap = len(claim_terms & terms) / max(1, len(claim_terms))
        
        if overlap > best_score:
            ctx_start = max(0, idx - 1)
            ctx_end = min(len(sentences), idx + 2)
            best_passage = " ".join(sentences[ctx_start:ctx_end])
            best_score = overlap
    
    return best_passage or text[:300].strip(), best_score


def _verify_claim_support(
    claim: str,
    passage: str,
    chunk_text: str,
    embedding_service: "EmbeddingService | None" = None
) -> tuple[str, float, str]:
    """
    Verify if claim is supported by the passage using semantic similarity.
    Returns: (status, score, reason)
    """
    if not passage.strip() and not chunk_text.strip():
        return "unsupported", 0.0, "No source content available."
    
    # Use embedding-based semantic verification if available
    if embedding_service is not None:
        try:
            claim_emb = embedding_service.create_embedding(claim)
            
            # Check both passage and full chunk
            passage_emb = embedding_service.create_embedding(passage)
            passage_sim = _cosine_similarity(claim_emb, passage_emb)
            
            chunk_sim = 0.0
            if chunk_text and chunk_text != passage:
                chunk_emb = embedding_service.create_embedding(chunk_text[:1000])  # Limit chunk size
                chunk_sim = _cosine_similarity(claim_emb, chunk_emb)
            
            # Use the higher similarity score
            semantic_score = max(passage_sim, chunk_sim)
            
            # Semantic similarity thresholds (well-calibrated for real embeddings)
            if semantic_score >= 0.70:
                return "supported", semantic_score, "Strong semantic alignment with source content."
            elif semantic_score >= 0.55:
                return "supported", semantic_score, "Good semantic match with source."
            elif semantic_score >= 0.40:
                return "partially_supported", semantic_score, "Moderate semantic relevance to source."
            elif semantic_score >= 0.25:
                return "partially_supported", semantic_score, "Weak semantic connection to source."
            else:
                return "unsupported", semantic_score, "Insufficient semantic alignment with source."
        
        except Exception:
            pass  # Fall through to lexical verification
    
    # Lexical fallback verification
    claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower())) - _STOPWORDS
    search_scope = f"{passage} {chunk_text}".lower()
    passage_terms = set(re.findall(r"[a-z0-9]+", search_scope)) - _STOPWORDS
    
    if not claim_terms:
        return "supported", 1.0, "Generic statement with no specific claims."
    
    lexical_coverage = len(claim_terms & passage_terms) / max(1, len(claim_terms))
    
    # Check for critical numeric mismatches
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    passage_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", search_scope))
    has_numeric_mismatch = bool(claim_numbers and (claim_numbers - passage_numbers))
    
    if has_numeric_mismatch:
        missing = ", ".join(sorted(claim_numbers - passage_numbers))
        if lexical_coverage >= 0.30:
            return "partially_supported", lexical_coverage, f"Topically relevant but numeric value '{missing}' not in source."
        else:
            return "unsupported", lexical_coverage, f"Key numeric value '{missing}' not found in source."
    
    # Lexical thresholds
    if lexical_coverage >= 0.35:
        return "supported", lexical_coverage, "Strong lexical overlap with source."
    elif lexical_coverage >= 0.20:
        return "supported", lexical_coverage, "Good lexical match with source."
    elif lexical_coverage >= 0.12:
        return "partially_supported", lexical_coverage, "Partial lexical overlap with source."
    else:
        return "unsupported", lexical_coverage, "Minimal lexical alignment with source."


def _origin(candidate: Candidate) -> str:
    parts = []
    if candidate.dense_rank is not None:
        parts.append(f"Dense #{candidate.dense_rank}")
    if candidate.bm25_rank is not None:
        parts.append(f"BM25 #{candidate.bm25_rank}")
    if candidate.rrf_score:
        parts.append(f"RRF {candidate.rrf_score:.4f}")
    return ", ".join(parts) or "Retrieved context"


def generate_claim_citations(
    answer: str,
    selected_candidates: list[Candidate],
    kb_version: int | str = 1,
    embedding_service: "EmbeddingService | None" = None
) -> tuple[list[Citation], CitationCoverage]:
    """
    Generate source-aware, semantically-verified citations.
    
    Extracts claims ONLY from the Answer section, ignoring Details and Sources.
    Uses embedding-based semantic similarity when available, falls back to
    lexical matching. Provides clear traceability from claims to sources.
    """
    claims = _split_into_claims(answer) if answer.strip() else []
    if not selected_candidates:
        return [], CitationCoverage(len(claims), 0, len(claims), 0.0, bool(claims), claims)
    
    citations: list[Citation] = []
    unsupported: list[str] = []
    fully_supported = partial = 0
    chunk_map = {index + 1: candidate for index, candidate in enumerate(selected_candidates)}
    
    for claim_id, claim in enumerate(claims, start=1):
        display_claim = re.sub(r"\[(?:Source\s*)?\d+\]", "", claim).strip()
        
        # Check for explicit source citations in the claim
        explicit = [int(v) for v in re.findall(r"\[(?:Source\s*)?(\d+)\]", claim) if int(v) in chunk_map]
        candidates = [chunk_map[index] for index in explicit] if explicit else selected_candidates
        
        # Find best supporting passage across candidates (semantic if possible)
        scored = [
            (_find_best_passage_semantic(display_claim, candidate.text or "", embedding_service), candidate)
            for candidate in candidates
        ]
        (passage, match_score), candidate = max(scored, key=lambda item: item[0][1])
        
        # Verify claim support with semantic or lexical analysis
        status, support_score, reason = _verify_claim_support(
            display_claim, passage, candidate.text or "", embedding_service
        )
        
        # Update counters
        if status == "supported":
            fully_supported += 1
        elif status == "partially_supported":
            partial += 1
        else:
            unsupported.append(display_claim)
        
        # Extract metadata
        meta = candidate.metadata if candidate else {}
        file_type = meta.get("file_type") or (
            Path(candidate.filename).suffix.lstrip(".").lower() 
            if candidate and candidate.filename else ""
        )
        page_num = meta.get("page_number")
        try:
            page_number = int(page_num) if page_num and int(page_num) > 0 else None
        except (ValueError, TypeError):
            page_number = None
        
        # Create citation
        citations.append(Citation(
            id=claim_id,
            marker=f"[{claim_id}]",
            claim=display_claim,
            filename=candidate.filename if status != "unsupported" or explicit else "None",
            chunk_id=candidate.chunk_id if status != "unsupported" or explicit else "N/A",
            passage=passage if passage else "[No supporting passage found]",
            retrieval_source=_origin(candidate) if status != "unsupported" or explicit else "Unverified",
            status=status,
            verification_reason=f"{status.replace('_', ' ').title()}: {reason} (score: {support_score:.2f})",
            kb_version=kb_version,
            support_score=support_score,
            lexical_overlap=match_score,
            direct_answer_support=status == "supported",
            file_type=file_type,
            page_number=page_number,
        ))
    
    # Calculate overall coverage
    total = len(claims)
    cov_pct = round(
        ((fully_supported * 1.0 + partial * 0.5) / total * 100.0) if total else 100.0,
        1
    )
    
    return citations, CitationCoverage(
        total_claims=total,
        supported_claims=fully_supported,
        unsupported_claims=len(unsupported),
        coverage_percentage=cov_pct,
        has_unsupported=bool(unsupported),
        unsupported_claims_list=unsupported,
        partially_supported_claims=partial,
    )


