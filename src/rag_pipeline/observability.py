"""Orchestrate Langfuse tracing and optional DeepEval evaluation per query."""
from __future__ import annotations

import logging
from typing import Any

from src.rag_pipeline.langfuse_tracing import emit_langfuse_trace
from src.rag_pipeline.observability_config import (
    deepeval_compare_all_retrievers,
    deepeval_enabled,
    deepeval_eval_modes,
    ragas_compare_all_retrievers,
    ragas_eval_enabled,
    ragas_eval_modes,
)

logger = logging.getLogger(__name__)


def _format_terminal_output(
    *,
    query: str,
    retrieval_mode: str,
    response: dict[str, Any],
    eval_result: dict[str, Any],
    trace_id: str | None,
    eval_type: str = "evaluation",
) -> None:
    """Print concise evaluation + Langfuse summary to terminal."""
    answer = response.get("raw_answer") or response.get("answer") or ""
    answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
    
    scores = eval_result.get("scores", {})
    status = eval_result.get("status", "unknown")
    reason = eval_result.get("reason", "")
    error = eval_result.get("error", "")
    
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print(f"Method: {retrieval_mode.upper()}")
    print("-" * 80)
    print(f"Answer: {answer_preview}")
    print("-" * 80)
    
    if status == "skipped":
        print(f"{eval_type.upper()}: Skipped ({reason})")
    elif status == "error":
        print(f"{eval_type.upper()}: Error - {error or reason}")
    elif scores:
        print(f"{eval_type.upper()} Metrics:")
        for metric_name, metric_data in scores.items():
            # Format metric name nicely
            display_name = metric_name.replace("_", " ").title()
            if isinstance(metric_data, dict):
                score = metric_data.get("score", 0)
                reason_text = metric_data.get("reason", "")
                print(f"  {display_name}: {score:.4f}")
                if reason_text:
                    print(f"    Reason: {reason_text[:100]}...")
            else:
                print(f"  {display_name}: {float(metric_data):.4f}")
    else:
        print(f"{eval_type.upper()}: Not evaluated")
    
    print("-" * 80)
    if trace_id:
        print(f"Langfuse Trace ID: {trace_id}")
    else:
        print("Langfuse: Not enabled")
    print("=" * 80 + "\n")


def observe_pipeline_answer(
    *,
    pipeline: Any,
    question: str,
    retrieval_mode: str,
    response: dict[str, Any],
    total_latency_ms: float,
    labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Attach Langfuse trace and optional DeepEval/RAGAS scores to a completed pipeline response.

    When DeepEval is enabled, runs evaluation on the actual pipeline response.
    When comparison is enabled, runs additional pipeline passes for other retrieval modes.
    """
    deepeval_by_mode: dict[str, dict[str, Any]] = {}
    ragas_by_mode: dict[str, dict[str, Any]] = {}
    timings = (response.get("trace") or {}).get("retrieval", {}).get("timings_ms") or {}

    # Use DeepEval if enabled, otherwise fall back to RAGAS
    use_deepeval = deepeval_enabled()

    if use_deepeval:
        # DeepEval evaluation path
        modes = deepeval_eval_modes()
        mode_responses: dict[str, dict[str, Any]] = {retrieval_mode: response}

        if deepeval_compare_all_retrievers():
            for mode in modes:
                if mode == retrieval_mode:
                    continue
                try:
                    mode_responses[mode] = pipeline._answer_question_impl(
                        question,
                        mode,
                    )
                except Exception:
                    logger.exception(
                        "Comparison pipeline run failed for mode=%s", mode
                    )

        for mode, mode_response in mode_responses.items():
            if mode not in modes:
                continue
            try:
                from src.rag_pipeline.deepeval_evaluation import evaluate_rag_response

                answer = mode_response.get("raw_answer") or mode_response.get("answer") or ""
                trace = mode_response.get("trace") or {}
                context_block = trace.get("context") or {}
                chunks = context_block.get("final_chunks") or mode_response.get("retrieved_chunks") or []
                contexts = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        text = chunk.get("text") or chunk.get("content") or ""
                        if text:
                            contexts.append(str(text))
                    elif chunk:
                        contexts.append(str(chunk))

                deepeval_by_mode[mode] = evaluate_rag_response(
                    query=question,
                    answer=answer,
                    retrieved_contexts=contexts,
                    retrieval_mode=mode,  # type: ignore[arg-type]
                )
            except Exception:
                logger.exception("DeepEval evaluation failed for mode=%s", mode)
                deepeval_by_mode[mode] = {
                    "status": "error",
                    "error": "Evaluation execution failed",
                    "scores": {},
                    "retrieval_mode": mode,
                }

        response["deepeval"] = {
            "by_retrieval_mode": deepeval_by_mode,
        }

    elif ragas_eval_enabled():
        # RAGAS evaluation path (legacy)
        from src.rag_pipeline.ragas_evaluation import evaluate_response_with_ragas

        modes = ragas_eval_modes()
        mode_responses: dict[str, dict[str, Any]] = {retrieval_mode: response}

        if ragas_compare_all_retrievers():
            for mode in modes:
                if mode == retrieval_mode:
                    continue
                try:
                    mode_responses[mode] = pipeline._answer_question_impl(
                        question,
                        mode,
                    )
                except Exception:
                    logger.exception(
                        "Comparison pipeline run failed for mode=%s", mode
                    )

        for mode, mode_response in mode_responses.items():
            if mode not in modes:
                continue
            ragas_by_mode[mode] = evaluate_response_with_ragas(
                query=question,
                response=mode_response,
                retrieval_mode=mode,  # type: ignore[arg-type]
                labels=labels,
            )

        response["ragas"] = {
            "by_retrieval_mode": ragas_by_mode,
            "metrics_plan": ragas_by_mode.get(retrieval_mode, {}).get("metrics_plan"),
        }

    trace_id = emit_langfuse_trace(
        query=question,
        retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
        response=response,
        total_latency_ms=total_latency_ms,
        retrieval_timings_ms=timings,
        deepeval_by_mode=deepeval_by_mode or None,
        ragas_by_mode=ragas_by_mode or None,
    )
    if trace_id:
        response.setdefault("observability", {})["langfuse_trace_id"] = trace_id

    # Print terminal output
    if use_deepeval and deepeval_enabled() and retrieval_mode in deepeval_by_mode:
        current_eval = deepeval_by_mode.get(retrieval_mode, {})
        _format_terminal_output(
            query=question,
            retrieval_mode=retrieval_mode,
            response=response,
            eval_result=current_eval,
            trace_id=trace_id,
            eval_type="deepeval",
        )
    elif ragas_eval_enabled() and retrieval_mode in ragas_by_mode:
        current_eval = ragas_by_mode.get(retrieval_mode, {})
        _format_terminal_output(
            query=question,
            retrieval_mode=retrieval_mode,
            response=response,
            eval_result=current_eval,
            trace_id=trace_id,
            eval_type="ragas",
        )

    return response
