from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st

from src.config import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR
from src.rag_pipeline import RAGPipeline
from src.ui.astronomy_theme import (
    inject_global_css,
    render_context_chunks,
    render_landing_page,
    render_panel_card,
    render_sources_card,
    render_stat_grid,
    render_status_row,
)


def _pipeline() -> RAGPipeline:
    """Return (and cache) a RAGPipeline in session_state."""
    if (
        "pipeline" not in st.session_state
        or st.session_state.get("force_reload", False)
    ):
        st.session_state.pipeline = RAGPipeline(
            knowledge_base_path=Path(KNOWLEDGE_BASE_DIR),
            vectorstore_path=Path(VECTORSTORE_DIR),
        )
        st.session_state.force_reload = False

    return st.session_state.pipeline


def _render_assistant_extras(
    sources: list[str],
    chunks: list[dict],
    num_retrieved: int | None = None,
) -> None:
    """Render sources and retrieved context below an assistant answer."""
    if sources:
        st.html(
            render_sources_card(
                sources,
                num_retrieved=num_retrieved,
            )
        )
    else:
        st.html(
            '<div class="meta-text">No relevant sources were found.</div>'
        )

    if chunks:
        with st.expander("View retrieved context"):
            st.html(render_context_chunks(chunks))


def _show_ingestion_notice() -> None:
    """Display a one-time ingestion result after a rerun."""
    notice = st.session_state.pop("ingestion_notice", None)

    if not notice:
        return

    notice_type = notice.get("type", "info")
    message = notice.get("message", "")

    if notice_type == "success":
        st.success(message)
    elif notice_type == "warning":
        st.warning(message)
    elif notice_type == "error":
        st.error(message)
    else:
        st.info(message)


def _render_custom_composer(pipeline: RAGPipeline) -> str | None:
    """
    Render the custom astronomy composer.

    The question input and send button are inside a Streamlit form, so
    pressing Enter submits the question just like clicking the arrow.
    A versioned widget key creates a fresh empty input after submission.
    """

    if "composer_version" not in st.session_state:
        st.session_state.composer_version = 0

    plus_col, composer_col = st.columns(
        [0.75, 10.25],
        gap="small",
    )

    # ------------------------------------------------------------
    # PLUS / KNOWLEDGE UPLOAD
    # ------------------------------------------------------------
    with plus_col:
        with st.popover("＋", help="Add knowledge to the knowledge base"):
            st.markdown(
                """
                <div class="add-file-title">
                    Add knowledge
                </div>
                <div class="add-file-subtitle">
                    Upload a TXT or Markdown file to expand the astronomy
                    knowledge base.
                </div>
                <div class="add-file-preserve">
                    ● Incremental indexing · Existing knowledge preserved
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded = st.file_uploader(
                "Choose a file",
                type=["txt", "md"],
                key="composer_file_uploader",
                label_visibility="collapsed",
            )

            if uploaded is not None:
                st.markdown(
                    f"""
                    <div class="selected-file">
                        <span class="selected-file-icon">◈</span>
                        <span>{html.escape(uploaded.name)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Add to Knowledge Base",
                    key="composer_add_file",
                    use_container_width=True,
                ):
                    with st.spinner(f"Ingesting {uploaded.name} …"):
                        try:
                            content = uploaded.getvalue().decode(
                                "utf-8",
                                errors="ignore",
                            )

                            result = pipeline.ingest_file(
                                file_path=uploaded.name,
                                content=content,
                            )

                            if result["skipped"]:
                                st.session_state.ingestion_notice = {
                                    "type": "info",
                                    "message": (
                                        f"{uploaded.name} is already "
                                        "up-to-date. No embeddings were "
                                        "changed."
                                    ),
                                }
                            else:
                                st.session_state.ingestion_notice = {
                                    "type": "success",
                                    "message": (
                                        f"{uploaded.name} added "
                                        f"successfully — "
                                        f"{result['chunks_added']} "
                                        "new chunks indexed. "
                                        f"Total indexed: "
                                        f"{result['collection_count']}."
                                    ),
                                }

                            st.rerun()

                        except Exception as exc:
                            st.error(f"Ingestion failed: {exc}")

    # ------------------------------------------------------------
    # QUESTION COMPOSER
    # ------------------------------------------------------------
    with composer_col:
        with st.form(
            key=f"question_form_{st.session_state.composer_version}",
            clear_on_submit=False,
            border=False,
        ):
            input_col, send_col = st.columns(
                [9.35, 0.65],
                gap="small",
            )

            with input_col:
                question = st.text_input(
                    "Ask a question",
                    placeholder="Ask the universe something…",
                    key=(
                        f"custom_question_"
                        f"{st.session_state.composer_version}"
                    ),
                    label_visibility="collapsed",
                )

            with send_col:
                send_clicked = st.form_submit_button(
                    "➤",
                    use_container_width=True,
                    help="Send question",
                )

    if send_clicked:
        question = question.strip()

        if not question:
            return None

        # Do not assign directly to the current widget's session-state key.
        # Instead, create a new widget key on the next rerun. This clears
        # the input without triggering Streamlit's widget-state exception.
        st.session_state.composer_version += 1

        return question

    return None


def main() -> None:
    st.set_page_config(
        page_title="Astronomy Knowledge Assistant",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.html(inject_global_css())

    dev_embeddings = (
        os.getenv("DEV_EMBEDDINGS", "false").lower()
        in ("1", "true", "yes")
    )

    Path(KNOWLEDGE_BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

    pipeline = _pipeline()
    status = pipeline.status()

    _show_ingestion_notice()

    try:
        actual_count = pipeline.vector_store.get_collection_count()
    except Exception:
        actual_count = 0

    with st.sidebar:
        core_html = render_stat_grid(
            [
                (str(actual_count), "Knowledge chunks"),
                (str(status["txt_files_found"]), "Source files"),
            ]
        )

        core_html += render_status_row(
            "Vector Index",
            status["index_available"],
        )
        core_html += render_status_row(
            "BM25",
            bool(status.get("bm25_available")),
        )
        core_html += render_status_row(
            "Reranker",
            bool(status.get("reranker_available")),
            warning=not status.get("reranker_available"),
        )

        st.html(
            render_panel_card(
                "🌌 Knowledge Core",
                core_html,
            )
        )

        meta_html = (
            '<div class="meta-text">'
            f'Embedding: {html.escape(str(status["embedding_model"]))}'
            "</div>"
        )

        if dev_embeddings:
            meta_html += (
                '<div class="meta-text">Mode: Synthetic (DEV)</div>'
            )

        meta_html += (
            '<div class="meta-text">'
            f'Store: {html.escape(str(status["vectorstore_path"]))}'
            "</div>"
        )

        st.html(
            render_panel_card(
                "📡 Knowledge Store",
                meta_html,
            )
        )

        st.html(render_panel_card("⚙ Index Control", ""))

        if st.button("Refresh Status", use_container_width=True):
            st.session_state.force_reload = True
            st.rerun()

        if st.button("Rebuild Full Index", use_container_width=True):
            with st.spinner("Embedding missing knowledge base chunks …"):
                try:
                    result = pipeline.ingest_documents()
                    st.session_state.force_reload = True

                    skipped = result.get("chunks_skipped", 0)

                    st.session_state.ingestion_notice = {
                        "type": "success",
                        "message": (
                            f"Index updated: "
                            f"{result['chunks_indexed']} new chunks "
                            f"({skipped} skipped, "
                            f"{result['collection_count']} total) "
                            f"from {result['files_processed']} files."
                        ),
                    }

                    st.rerun()

                except Exception as exc:
                    st.error(f"Rebuild failed: {exc}")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.pop("chat_history", None)
            st.rerun()

    if dev_embeddings:
        st.html(
            '<div class="dev-banner">'
            "⚠ DEV MODE: Synthetic embeddings are active. "
            "Semantic retrieval will not reflect real content similarity."
            "</div>"
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    has_chat = bool(st.session_state.chat_history)

    if not has_chat:
        st.html(
            render_landing_page(
                index_ready=status["index_available"],
                rag_ready=status["index_available"],
            )
        )

        if not status["index_available"]:
            st.html(
                '<div class="warning-banner">'
                "No vector index found. "
                "Use <strong>Rebuild Full Index</strong> "
                "in the sidebar, or run: "
                "<code>python ingest.py</code>"
                "</div>"
            )

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

            if entry["role"] == "assistant":
                _render_assistant_extras(
                    sources=entry.get("sources", []),
                    chunks=entry.get("retrieved_chunks", []),
                    num_retrieved=(
                        len(entry.get("retrieved_chunks", [])) or None
                    ),
                )

    question = _render_custom_composer(pipeline)

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Searching the astronomy knowledge base …"):
            try:
                response = pipeline.answer_question(question)
            except Exception as exc:
                response = {
                    "answer": "I couldn't process that question right now.",
                    "sources": [],
                    "retrieved_chunks": [],
                    "error": str(exc),
                }

        answer = response.get("answer", "No answer available.")
        sources = response.get("sources", [])
        chunks = response.get("retrieved_chunks", [])

        with st.chat_message("assistant"):
            st.markdown(answer)
            _render_assistant_extras(
                sources=sources,
                chunks=chunks,
                num_retrieved=response.get("num_retrieved"),
            )

        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "retrieved_chunks": chunks,
            }
        )

        st.rerun()


if __name__ == "__main__":
    main()
