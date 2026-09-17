from __future__ import annotations

import re
from typing import Any, Dict, Optional


def normalize_text(text: Any) -> str:
    """
    Normalize text for comparison, matching, and diagnostics.

    Keeps the actual semantic content while removing excessive
    whitespace and normalizing case.
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """
    try:
        if value is None:
            return default

        result = float(value)

        if result != result:  # NaN
            return default

        return result

    except (TypeError, ValueError):
        return default


def candidate_key(
    candidate_or_filename: Any,
    chunk_id: Optional[Any] = None,
) -> str:
    """
    Build a stable identifier for a retrieved chunk.

    Supports both calling styles used by the pipeline:

        candidate_key(candidate_dict)

    and:

        candidate_key(filename, chunk_id)

    This compatibility is important because retrieval.py uses:

        candidate_key(filename, chunk_id)

    while other parts of the pipeline may pass a candidate dictionary.
    """

    # Style 1:
    # candidate_key(filename, chunk_id)
    if chunk_id is not None:
        filename = str(candidate_or_filename or "")
        return f"{filename}::chunk-{chunk_id}"

    # Style 2:
    # candidate_key(candidate_dict)
    if isinstance(candidate_or_filename, dict):
        filename = str(
            candidate_or_filename.get("filename", "")
        )

        cid = candidate_or_filename.get(
            "chunk_id",
            "",
        )

        return f"{filename}::chunk-{cid}"

    # Fallback for a plain string/object.
    return str(candidate_or_filename or "")


def get_retrieval_mode(
    mode: Optional[str] = None,
) -> str:
    """
    Normalize the retrieval mode.

    Supported modes:
        vector / dense
        bm25
        hybrid / both / rrf

    Hybrid is the default.
    """

    if mode is None:
        return "hybrid"

    normalized = normalize_text(mode)

    if normalized in {
        "vector",
        "dense",
        "vector only",
        "dense only",
        "vector-only",
        "dense-only",
    }:
        return "vector"

    if normalized in {
        "bm25",
        "keyword",
        "sparse",
        "bm25 only",
        "keyword only",
        "sparse only",
        "bm25-only",
    }:
        return "bm25"

    if normalized in {
        "hybrid",
        "both",
        "dense + bm25",
        "bm25 + dense",
        "dense and bm25",
        "bm25 and dense",
        "rrf",
    }:
        return "hybrid"

    # Unknown mode → safe default.
    return "hybrid"


def is_greeting(query: str) -> bool:
    """
    Detect simple conversational greetings so they don't trigger
    unnecessary knowledge-base retrieval.
    """

    text = normalize_text(query)

    if not text:
        return False

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "bye",
        "goodbye",
    }

    if text in greetings:
        return True

    # Small variations such as "hi there".
    greeting_patterns = [
        r"^hi there$",
        r"^hello there$",
        r"^hey there$",
        r"^hey everyone$",
        r"^hello everyone$",
        r"^hi everyone$",
    ]

    return any(
        re.fullmatch(pattern, text)
        for pattern in greeting_patterns
    )


def is_multi_intent_query(query: str) -> bool:
    """
    Detect whether a query appears to contain multiple distinct
    information requests.

    This is intentionally conservative. It is used for diagnostics
    and query preparation, not as a hard retrieval filter.
    """

    text = normalize_text(query)

    if not text:
        return False

    # Explicit conjunctions often indicate multiple topics.
    separators = [
        " and ",
        " also ",
        " as well as ",
        " along with ",
        " plus ",
    ]

    separator_hits = sum(
        text.count(separator)
        for separator in separators
    )

    # Multiple question marks are a strong signal.
    question_count = text.count("?")

    # Comma-separated long queries can sometimes contain multiple
    # intents, but we avoid treating every comma as multi-intent.
    comma_count = text.count(",")

    if question_count >= 2:
        return True

    if separator_hits >= 1 and len(text.split()) >= 6:
        return True

    if comma_count >= 2 and len(text.split()) >= 10:
        return True

    return False


def prepare_search_query(
    query: str,
) -> Dict[str, Any]:
    """
    Prepare a query for retrieval while preserving the original
    user wording.

    The original query is always retained.

    Returns:
        {
            "original_query": ...,
            "search_query": ...,
            "normalized_query": ...,
            "is_greeting": ...,
            "is_multi_intent": ...
        }
    """

    original_query = str(query or "").strip()
    normalized_query = normalize_text(original_query)

    greeting = is_greeting(original_query)
    multi_intent = is_multi_intent_query(original_query)

    # IMPORTANT:
    # Do not aggressively rewrite the user's query.
    # Dense and BM25 retrieval should see the actual semantic query.
    search_query = original_query

    return {
        "original_query": original_query,
        "search_query": search_query,
        "normalized_query": normalized_query,
        "is_greeting": greeting,
        "is_multi_intent": multi_intent,
    }


def tokenize_query(query: str) -> list[str]:
    """
    Tokenize a query using the same simple lexical style used by
    the BM25 retriever.
    """

    text = normalize_text(query)

    if not text:
        return []

    return re.findall(
        r"[a-z0-9]+",
        text,
    )


def get_query_diagnostics(
    query: str,
    retrieval_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return runtime diagnostics describing how the query will be
    processed.

    This function does not perform retrieval.
    """

    prepared = prepare_search_query(query)

    tokens = tokenize_query(
        prepared["search_query"]
    )

    mode = get_retrieval_mode(
        retrieval_mode
    )

    return {
        "original_query": prepared["original_query"],
        "normalized_query": prepared["normalized_query"],
        "search_query": prepared["search_query"],
        "retrieval_mode": mode,
        "token_count": len(tokens),
        "tokens": tokens,
        "is_greeting": prepared["is_greeting"],
        "is_multi_intent": prepared["is_multi_intent"],
    }
