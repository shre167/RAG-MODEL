"""Environment-driven settings for Langfuse and RAGAS (no secrets in code)."""
from __future__ import annotations

import os
from typing import Literal

RetrievalMode = Literal["bm25", "vector", "hybrid"]

_DEFAULT_RAGAS_MODES = ("bm25", "vector", "hybrid")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def langfuse_enabled() -> bool:
    if _env_flag("DISABLE_LANGFUSE", default=False):
        return False
    if not _env_flag("ENABLE_LANGFUSE", default=True):
        return False
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    )


def ragas_eval_enabled() -> bool:
    return _env_flag("ENABLE_RAGAS_EVAL", default=False)


def ragas_eval_modes() -> tuple[RetrievalMode, ...]:
    raw = os.getenv("RAGAS_EVAL_MODES", ",".join(_DEFAULT_RAGAS_MODES))
    modes: list[RetrievalMode] = []
    for part in raw.split(","):
        mode = part.strip().lower()
        if mode in _DEFAULT_RAGAS_MODES:
            modes.append(mode)  # type: ignore[arg-type]
    return tuple(modes) if modes else _DEFAULT_RAGAS_MODES


def ragas_compare_all_retrievers() -> bool:
    """When true, run the pipeline once per retrieval mode for RAGAS comparison."""
    return _env_flag("RAGAS_COMPARE_ALL_RETRIEVERS", default=ragas_eval_enabled())


def deepeval_enabled() -> bool:
    """Check if DeepEval evaluation is enabled."""
    return _env_flag("ENABLE_DEEPEVAL", default=False)


def deepeval_compare_all_retrievers() -> bool:
    """When true, run the pipeline once per retrieval mode for DeepEval comparison."""
    return _env_flag("DEEPEVAL_COMPARE_ALL_RETRIEVERS", default=deepeval_enabled())


def deepeval_eval_modes() -> tuple[RetrievalMode, ...]:
    """Get the retrieval modes to evaluate with DeepEval."""
    raw = os.getenv("DEEPEVAL_EVAL_MODES", ",".join(_DEFAULT_RAGAS_MODES))
    modes: list[RetrievalMode] = []
    for part in raw.split(","):
        mode = part.strip().lower()
        if mode in _DEFAULT_RAGAS_MODES:
            modes.append(mode)  # type: ignore[arg-type]
    return tuple(modes) if modes else _DEFAULT_RAGAS_MODES
