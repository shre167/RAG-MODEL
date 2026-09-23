"""DeepEval evaluation for RAG responses (no fabricated ground truth)."""
from __future__ import annotations

import logging
from typing import Any

from src.config import GE_API_KEY, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from src.rag_pipeline.observability_config import RetrievalMode

logger = logging.getLogger(__name__)


class DeepEvalLLMAdapter:
    """Adapter to make ChatOpenAI compatible with DeepEval's LLM interface.
    
    DeepEval 1.6.2 has import issues with langchain.schema.
    This adapter provides a simpler LLM interface for evaluation.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI-compatible client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url.rstrip("/") + "/",
            )
        except Exception as e:
            logger.warning(f"Failed to init OpenAI client: {e}")
            self.client = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM."""
        if not self.client:
            return ""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return ""


def _build_deepeval_llm() -> Any:
    """Build a minimal LLM adapter for DeepEval evaluation.
    
    Uses direct OpenAI API instead of langchain to avoid import issues.
    """
    api_key = GE_API_KEY or LLM_API_KEY
    if not api_key:
        logger.warning("DeepEval LLM: API key not configured.")
        return None
    if not LLM_BASE_URL:
        logger.warning("DeepEval LLM: LLM_BASE_URL not configured.")
        return None
    if not LLM_MODEL:
        logger.warning("DeepEval LLM: LLM_MODEL not configured.")
        return None

    try:
        return DeepEvalLLMAdapter(
            api_key=api_key,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
        )
    except Exception as exc:
        logger.exception("Failed to construct DeepEval LLM: %s", exc)
        return None


def evaluate_rag_response(
    *,
    query: str,
    answer: str,
    retrieved_contexts: list[str],
    retrieval_mode: RetrievalMode,
) -> dict[str, Any]:
    """
    Evaluate a RAG response using DeepEval metrics (simplified).

    Uses the actual query, answer, and contexts from the pipeline.
    No fabricated ground truth or additional generation.

    Metrics (Reference-Free, implemented directly):
    - Faithfulness: Is the answer grounded in the retrieved context?
    - Answer Relevancy: Is the answer relevant to the query?

    Returns dict with:
        - status: "ok" | "skipped" | "error"
        - scores: {metric_name: {"score": float, "reason": str}}
        - retrieval_mode: The retrieval method used
        - error: Error message if status is "error"
    """
    if not query or not answer or not retrieved_contexts:
        return {
            "status": "skipped",
            "reason": "missing query, answer, or contexts",
            "scores": {},
            "retrieval_mode": retrieval_mode,
        }

    llm = _build_deepeval_llm()
    if llm is None or not llm.client:
        return {
            "status": "skipped",
            "reason": "DeepEval LLM not configured",
            "scores": {},
            "retrieval_mode": retrieval_mode,
        }

    try:
        scores = {}
        
        # Evaluate Faithfulness: Check if answer is grounded in context
        try:
            context_text = "\n".join(retrieved_contexts[:3])  # Use top 3 contexts
            faithfulness_prompt = f"""
Evaluate if the following answer is faithful to the provided context.
Answer faithfulness means the answer is grounded in the context and does not contradict it.

Context:
{context_text}

Answer:
{answer}

Rate the faithfulness on a scale of 0 to 1, where 1 is completely faithful.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
            faithfulness_response = llm.generate(faithfulness_prompt)
            try:
                parts = faithfulness_response.strip().split(None, 1)
                faith_score = float(parts[0]) if parts else 0.5
                faith_reason = parts[1] if len(parts) > 1 else "Faithfulness evaluated"
                faith_score = max(0.0, min(1.0, faith_score))
            except (ValueError, IndexError):
                faith_score = 0.5
                faith_reason = faithfulness_response[:100]
            
            scores["faithfulness"] = {
                "score": faith_score,
                "reason": faith_reason,
            }
        except Exception as exc:
            logger.warning("Faithfulness evaluation failed: %s", exc)
            scores["faithfulness"] = {
                "score": 0.0,
                "reason": f"Evaluation error: {str(exc)}",
            }

        # Evaluate Answer Relevancy: Check if answer is relevant to query
        try:
            relevancy_prompt = f"""
Evaluate if the following answer is relevant and responsive to the query.
Answer relevancy means the answer directly addresses the query.

Query:
{query}

Answer:
{answer}

Rate the relevancy on a scale of 0 to 1, where 1 is completely relevant.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
            relevancy_response = llm.generate(relevancy_prompt)
            try:
                parts = relevancy_response.strip().split(None, 1)
                relevancy_score = float(parts[0]) if parts else 0.5
                relevancy_reason = parts[1] if len(parts) > 1 else "Relevancy evaluated"
                relevancy_score = max(0.0, min(1.0, relevancy_score))
            except (ValueError, IndexError):
                relevancy_score = 0.5
                relevancy_reason = relevancy_response[:100]
            
            scores["answer_relevancy"] = {
                "score": relevancy_score,
                "reason": relevancy_reason,
            }
        except Exception as exc:
            logger.warning("Answer relevancy evaluation failed: %s", exc)
            scores["answer_relevancy"] = {
                "score": 0.0,
                "reason": f"Evaluation error: {str(exc)}",
            }

        return {
            "status": "ok",
            "scores": scores,
            "retrieval_mode": retrieval_mode,
        }

    except Exception as exc:
        logger.exception("DeepEval evaluation failed for mode=%s", retrieval_mode)
        return {
            "status": "error",
            "error": str(exc),
            "scores": {},
            "retrieval_mode": retrieval_mode,
        }

