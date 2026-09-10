from __future__ import annotations

import logging
import random
import time
from typing import Callable, List, Optional, TypeVar

import requests

try:
    from openai import OpenAI as OpenAIClient, APIStatusError
    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    OpenAIClient = None  # type: ignore
    APIStatusError = Exception  # type: ignore
    _HAS_OPENAI = False

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_BASE_DELAY,
    GE_API_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimitError(RuntimeError):
    """Raised when embedding API rate limits are exceeded after all retries."""


def _is_rate_limit_error(exc: Exception) -> bool:
    if _HAS_OPENAI and isinstance(exc, APIStatusError):
        if getattr(exc, "status_code", None) == 429:
            return True
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        if response is not None and response.status_code == 429:
            return True
    message = str(exc).lower()
    return (
        "429" in message
        or "rate limit" in message
        or "quota" in message
        or "resource exhausted" in message
        or "too many requests" in message
    )


def _get_retry_after_seconds(exc: Exception) -> Optional[float]:
    if _HAS_OPENAI and isinstance(exc, APIStatusError):
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except (TypeError, ValueError):
                    pass

    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("Retry-After") or response.headers.get(
                "retry-after"
            )
            if retry_after is not None:
                try:
                    return float(retry_after)
                except (TypeError, ValueError):
                    pass
    return None


def _execute_with_retry(
    operation_name: str,
    func: Callable[[], T],
    *,
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
) -> T:
    retries = max_retries if max_retries is not None else EMBEDDING_MAX_RETRIES
    delay_base = base_delay if base_delay is not None else EMBEDDING_RETRY_BASE_DELAY
    last_exc: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise
            last_exc = exc
            if attempt >= retries:
                break
            retry_after = _get_retry_after_seconds(exc)
            delay = retry_after if retry_after is not None else delay_base * (2 ** attempt)
            delay += random.uniform(0, 0.5)
            logger.warning(
                "Rate limit (429) on %s, attempt %d/%d. Retrying in %.1fs.",
                operation_name,
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)

    raise RateLimitError(
        f"Rate limit exceeded after {retries} retries during {operation_name}: {last_exc}"
    ) from last_exc


class EmbeddingService:
    """Create embeddings using an OpenAI-compatible endpoint.

    Uses the OpenAI SDK client when available (handles Bearer auth cleanly).
    Falls back to a raw requests.post() call ONLY when the OpenAI client
    itself cannot be initialised (not when an API call fails with a 4xx).

    This eliminates the previous double-request pattern where a failed
    client call fell through to a redundant requests.post() with the same
    payload, causing every error to generate two network round-trips.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or GE_API_KEY or LLM_API_KEY
        self.model = model or EMBEDDING_MODEL
        self.base_url = (base_url or LLM_BASE_URL or "").rstrip("/")

        self._client: Optional["OpenAIClient"] = None
        if _HAS_OPENAI and self.api_key and self.base_url:
            try:
                try:
                    self._client = OpenAIClient(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        default_headers={"x-api-key": self.api_key},
                    )
                except TypeError:
                    # older openai SDK versions don't support default_headers
                    self._client = OpenAIClient(
                        api_key=self.api_key,
                        base_url=self.base_url,
                    )
            except Exception:
                self._client = None

    # ------------------------------------------------------------------
    # Single-text embedding
    # ------------------------------------------------------------------

    def create_embedding(self, text: str) -> Optional[List[float]]:
        if not self.base_url or not self.model or not self.api_key:
            logger.warning(
                "Embedding config incomplete: GE_API_KEY, LLM_BASE_URL, "
                "and EMBEDDING_MODEL must all be set in .env."
            )
            return None

        if self._client is not None:
            try:
                resp = _execute_with_retry(
                    "single embedding",
                    lambda: self._client.embeddings.create(  # type: ignore[union-attr]
                        model=self.model,
                        input=text,
                    ),
                )
                return resp.data[0].embedding
            except RateLimitError:
                raise
            except Exception as exc:
                # Do NOT fall through to requests. If the client is configured
                # and the API returned an error, re-trying with raw requests
                # will produce the same failure. Surface the error clearly.
                logger.error(
                    "OpenAI client embedding error (model=%s): %s",
                    self.model,
                    exc,
                )
                return None

        # -- requests-only path (when OpenAI SDK is not installed) --
        return self._requests_embed_single(text)

    # ------------------------------------------------------------------
    # Batch embedding
    # ------------------------------------------------------------------

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.base_url or not self.model or not self.api_key:
            raise ValueError(
                "Embedding config incomplete: GE_API_KEY, LLM_BASE_URL, "
                "and EMBEDDING_MODEL must all be set in .env."
            )

        if self._client is not None:
            return self._client_embed_batch(texts)

        # -- requests-only fallback --
        return self._requests_embed_batch(texts)

    # ------------------------------------------------------------------
    # OpenAI SDK batch helper
    # ------------------------------------------------------------------

    def _client_embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding via OpenAI SDK with 429 retry/backoff."""
        try:
            resp = _execute_with_retry(
                f"batch embedding ({len(texts)} items)",
                lambda: self._client.embeddings.create(  # type: ignore[union-attr]
                    model=self.model,
                    input=texts,
                ),
            )
            results = [None] * len(texts)
            for item in resp.data:
                if item.index is not None and 0 <= item.index < len(texts):
                    results[item.index] = item.embedding
            none_indices = [i for i, value in enumerate(results) if value is None]
            if none_indices:
                logger.warning(
                    "Batch embed: %d items had no index; falling back per-item.",
                    len(none_indices),
                )
                for i in none_indices:
                    emb = self.create_embedding(texts[i])
                    if emb is None:
                        raise RuntimeError(
                            f"Embedding failed for item {i}/{len(texts)}"
                        )
                    results[i] = emb
            return results  # type: ignore[return-value]

        except RateLimitError:
            raise
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(
                    f"Rate limit exceeded during batch embedding: {exc}"
                ) from exc
            logger.warning(
                "Batch embedding failed (%s); retrying per-item.", exc
            )
            embeddings: List[List[float]] = []
            for i, text in enumerate(texts):
                emb = self.create_embedding(text)
                if emb is None:
                    raise RuntimeError(
                        f"Embedding failed for item {i + 1}/{len(texts)}"
                    ) from exc
                embeddings.append(emb)
                if (i + 1) % 10 == 0:
                    logger.info(
                        "Progress: %d / %d embeddings created.", i + 1, len(texts)
                    )
            return embeddings

    # ------------------------------------------------------------------
    # Raw requests helpers (no OpenAI SDK)
    # ------------------------------------------------------------------

    def _embeddings_endpoint(self) -> str:
        return f"{self.base_url}/embeddings"

    def _requests_embed_single(self, text: str) -> Optional[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": text}
        try:
            resp = _execute_with_retry(
                "single embedding (requests)",
                lambda: self._post_embedding_request(headers, payload, timeout=30),
            )
            return resp.json()["data"][0]["embedding"]
        except RateLimitError:
            raise
        except Exception as exc:
            logger.error("Embedding request failed: %s", exc)
            return None

    def _requests_embed_batch(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        try:
            resp = _execute_with_retry(
                f"batch embedding (requests, {len(texts)} items)",
                lambda: self._post_embedding_request(
                    headers,
                    {"model": self.model, "input": texts},
                    timeout=60,
                ),
            )
            items = resp.json().get("data", [])
            if len(items) != len(texts):
                raise RuntimeError(
                    f"Batch response length {len(items)} != request size {len(texts)}"
                )
            return [item["embedding"] for item in items]
        except RateLimitError:
            raise
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(
                    f"Rate limit exceeded during batch embedding: {exc}"
                ) from exc
            logger.warning("Batch embed (requests) failed (%s); retrying per-item.", exc)
            results: List[List[float]] = []
            for i, text in enumerate(texts):
                emb = self._requests_embed_single(text)
                if emb is None:
                    raise RuntimeError(
                        f"Embedding failed for item {i + 1}/{len(texts)}"
                    ) from exc
                results.append(emb)
            return results

    def _post_embedding_request(
        self,
        headers: dict[str, str],
        payload: dict,
        *,
        timeout: int,
    ) -> requests.Response:
        resp = requests.post(
            self._embeddings_endpoint(),
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            if resp.status_code == 429:
                raise exc
            raise
        return resp


class SyntheticEmbeddingService:
    """Deterministic synthetic embeddings for local development/testing without API calls."""

    DIMS = 768

    def __init__(self) -> None:
        self.model = "synthetic"
        logger.info("SyntheticEmbeddingService active — no API calls will be made.")

    def create_embedding(self, text: str) -> List[float]:
        import hashlib
        import math
        seed = int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest(), 16)
        vals: List[float] = []
        for i in range(self.DIMS):
            angle = (seed + i * 31337) % 360
            vals.append(math.sin(math.radians(angle)))
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.create_embedding(t) for t in texts]
