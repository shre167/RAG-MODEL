from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError

from src.config import (
    GE_API_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)
from src.rag_pipeline.config import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ======================================================================
# LLM GENERATION
# ======================================================================


def ask_llm(
    question: str,
    context: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    trace: dict[str, Any] | None = None,
) -> str:
    """
    Send the question and retrieved context to the LLM.

    Uses the OpenAI-compatible Generative Engine endpoint
    (https://openai.generative.engine.capgemini.com/v1) via the
    official OpenAI SDK. Credentials are never logged or stored.
    """
    resolved_api_key = (
        api_key
        or GE_API_KEY
        or LLM_API_KEY
    )

    resolved_model = (
        model_name
        or LLM_MODEL
    )

    resolved_base_url = (
        base_url
        or LLM_BASE_URL
    )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    if not resolved_api_key:
        logger.error("LLM API key is not configured.")
        return (
            "I could not generate the answer "
            "because the language-model API "
            "configuration is incomplete."
        )

    if not resolved_base_url:
        logger.error("LLM base URL is not configured.")
        return (
            "I could not generate the answer "
            "because the language-model endpoint "
            "is not configured."
        )

    if not resolved_model:
        logger.error("LLM model is not configured.")
        return (
            "I could not generate the answer "
            "because the language-model model "
            "is not configured."
        )

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    question_lower = question.lower()

    is_explanation = any(
        word in question_lower
        for word in (
            "explain",
            "describe",
            "why",
            "difference",
            "compare",
            "how does",
        )
    )

    is_procedural = any(
        phrase in question_lower
        for phrase in (
            "how to",
            "how do i",
            "steps",
            "procedure",
            "configure",
            "setup",
            "set up",
            "install",
        )
    )

    # ------------------------------------------------------------------
    # Build LLM instruction
    # ------------------------------------------------------------------

    user_instruction = (
        "Question:\n"
        f"{question}\n\n"
        "Retrieved Information:\n"
        f"{context}\n\n"
        "Answer the question using ONLY "
        "the retrieved information above.\n\n"
        "Do not use outside knowledge.\n\n"
        "If the retrieved information is "
        "insufficient, explicitly say that "
        "the knowledge base does not contain "
        "enough information to answer.\n\n"
        "CITATION REQUIREMENT:\n"
        "Ground every factual statement with a bracketed citation marker "
        "pointing to the supporting source chunk (for example: [1] for Source 1, "
        "[2] for Source 2). When an answer contains multiple factual claims, "
        "support each claim individually where practical.\n\n"
        "Only use source filenames that appear "
        "in the retrieved information.\n\n"
        "Include the source filenames under "
        "Sources."
    )

    if is_explanation:
        user_instruction += (
            "\nFor explanations, organize "
            "the answer clearly and use "
            "bullet points where useful."
        )

    if is_procedural:
        user_instruction += (
            "\nFor procedures, use numbered "
            "steps and include prerequisites "
            "only when they are supported by "
            "the retrieved information."
        )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_instruction,
        },
    ]

    # ------------------------------------------------------------------
    # Observatory trace
    # ------------------------------------------------------------------

    if trace is not None:
        trace.clear()

        trace.update(
            {
                "stage": "generation",
                "model": resolved_model,
                "base_url": resolved_base_url,
                "question": question,
                "system_prompt": SYSTEM_PROMPT,
                "user_instruction": user_instruction,
                "context": context,
                "context_chars": len(context),
                "message_count": len(messages),
                "temperature": 0.1,
                "explanation_request": is_explanation,
                "procedural_request": is_procedural,
                "status": "ready",
            }
        )

    # ------------------------------------------------------------------
    # LLM request (OpenAI-compatible interface)
    # ------------------------------------------------------------------

    try:
        client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=30,
        )

        logger.info(f"Calling OpenAI-compatible endpoint with model: {resolved_model}")

        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            max_completion_tokens=2048,
            temperature=0.1,
        )

        # --------------------------------------------------------------
        # Validate response
        # --------------------------------------------------------------

        if not response.choices:
            logger.error("Language model returned no choices.")

            if trace is not None:
                trace["status"] = "empty_choices"

            return "The language model returned no response."

        content = response.choices[0].message.content

        if not content:
            logger.error("Language model returned empty content.")

            if trace is not None:
                trace["status"] = "empty_content"

            return "The language model returned an empty response."

        answer = content.strip()

        if trace is not None:
            trace.update(
                {
                    "status": "success",
                    "response_chars": len(answer),
                    "response_preview": answer[:500],
                }
            )

        logger.info(f"Successfully received response: {len(answer)} characters")

        return answer

    except APITimeoutError:
        logger.error("Request to language model timed out after 30 seconds")

        if trace is not None:
            trace["status"] = "timeout"

        return "The language model request timed out. Please try again."

    except APIStatusError as status_exc:
        logger.error(
            f"Language model API returned status {status_exc.status_code}: {status_exc.message}"
        )

        if trace is not None:
            trace["status"] = f"api_error_{status_exc.status_code}"

        return (
            "The language model request failed. "
            "Please check the model configuration and connectivity."
        )

    except APIConnectionError as conn_exc:
        logger.error(f"Network error calling language model: {conn_exc}")

        if trace is not None:
            trace["status"] = "network_error"

        return (
            "A network error occurred while calling the language model. "
            "Please check your connection and try again."
        )

    except Exception as exc:
        import traceback

        print("\n========== LLM ERROR ==========")
        traceback.print_exc()
        print("MODEL =", resolved_model)
        print("BASE_URL =", resolved_base_url)
        print("ERROR =", repr(exc))
        print("================================\n")

        if trace is not None:
            trace["status"] = "exception"
            trace["error"] = str(exc)

        raise