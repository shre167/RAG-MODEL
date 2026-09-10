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
    if sources:
        st.markdown(
            render_sources_card(
                sources,
                num_retrieved=num_retrieved,
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="meta-text">No relevant sources were found.</div>',
            unsafe_allow_html=True,
        )

    if chunks:
        with st.expander("View retrieved context"):
            st.markdown(
                render_context_chunks(chunks),
                unsafe_allow_html=True,
            )


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
    Render a ChatGPT-style custom composer.

    The text field and send button are custom Streamlit controls.
    The + button opens the file uploader.
    """

    composer_col, send_col = st.columns(
        [11, 1],
        gap="small",
    )

    with composer_col:
        plus_col, input_col = st.columns(
            [0.75, 9.25],
            gap="small",
        )

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
                        with st.spinner(
                            f"Ingesting {uploaded.name} …"
                        ):
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
                                            f"new chunks indexed. "
                                            f"Total indexed: "
                                            f"{result['collection_count']}."
                                        ),
                                    }

                                st.rerun()

                            except Exception as exc:
                                st.error(
                                    f"Ingestion failed: {exc}"
                                )

        with input_col:
            question = st.text_input(
                "Ask a question",
                placeholder="Ask the universe something…",
                key="custom_question",
                label_visibility="collapsed",
            )

    with send_col:
        send_clicked = st.button(
            "➤",
            key="send_question",
            help="Send question",
            use_container_width=True,
        )

    if send_clicked and question.strip():
        return question.strip()

    return None


def main() -> None:
    st.set_page_config(
        page_title="Astronomy Knowledge Assistant",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        inject_global_css(),
        unsafe_allow_html=True,
    )

    dev_embeddings = (
        os.getenv("DEV_EMBEDDINGS", "false").lower()
        in ("1", "true", "yes")
    )

    Path(KNOWLEDGE_BASE_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(VECTORSTORE_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = _pipeline()
    status = pipeline.status()

    _show_ingestion_notice()

    try:
        actual_count = pipeline.vector_store.get_collection_count()
    except Exception:
        actual_count = 0

    # ================================================================
    # SIDEBAR
    # ================================================================

    with st.sidebar:
        core_html = render_stat_grid(
            [
                (str(actual_count), "Knowledge chunks"),
                (
                    str(status["txt_files_found"]),
                    "Source files",
                ),
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

        st.markdown(
            render_panel_card(
                "🌌 Knowledge Core",
                core_html,
            ),
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------
        # INDEX INFORMATION
        # ------------------------------------------------------------

        meta_html = (
            f'<div class="meta-text">'
            f'Embedding: {html.escape(str(status["embedding_model"]))}'
            f"</div>"
        )

        if dev_embeddings:
            meta_html += (
                '<div class="meta-text">'
                "Mode: Synthetic (DEV)"
                "</div>"
            )

        meta_html += (
            f'<div class="meta-text">'
            f'Store: {html.escape(str(status["vectorstore_path"]))}'
            f"</div>"
        )

        st.markdown(
            render_panel_card(
                "📡 Knowledge Store",
                meta_html,
            ),
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------
        # INDEX CONTROL
        # ------------------------------------------------------------

        st.markdown(
            render_panel_card(
                "⚙ Index Control",
                "",
            ),
            unsafe_allow_html=True,
        )

        if st.button(
            "Refresh Status",
            use_container_width=True,
        ):
            st.session_state.force_reload = True
            st.rerun()

        if st.button(
            "Rebuild Full Index",
            use_container_width=True,
        ):
            with st.spinner(
                "Embedding missing knowledge base chunks …"
            ):
                try:
                    result = pipeline.ingest_documents()

                    st.session_state.force_reload = True

                    skipped = result.get(
                        "chunks_skipped",
                        0,
                    )

                    st.success(
                        f"Index updated: "
                        f"{result['chunks_indexed']} new chunks "
                        f"({skipped} skipped, "
                        f"{result['collection_count']} total) "
                        f"from {result['files_processed']} files."
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"Rebuild failed: {exc}"
                    )

        if st.button(
            "Clear Chat",
            use_container_width=True,
        ):
            st.session_state.pop(
                "chat_history",
                None,
            )
            st.rerun()

    # ================================================================
    # DEV MODE
    # ================================================================

    if dev_embeddings:
        st.markdown(
            '<div class="dev-banner">'
            '⚠ DEV MODE: Synthetic embeddings are active. '
            "Semantic retrieval will not reflect real content similarity."
            "</div>",
            unsafe_allow_html=True,
        )

    # ================================================================
    # CHAT HISTORY
    # ================================================================

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    has_chat = bool(
        st.session_state.chat_history
    )

    if not has_chat:
        st.markdown(
            render_landing_page(
                index_ready=status["index_available"],
                rag_ready=status["index_available"],
            ),
            unsafe_allow_html=True,
        )

        if not status["index_available"]:
            st.markdown(
                '<div class="warning-banner">'
                "No vector index found. "
                "Use <strong>Rebuild Full Index</strong> "
                "in the sidebar, or run: "
                "<code>python ingest.py</code>"
                "</div>",
                unsafe_allow_html=True,
            )

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

            if entry["role"] == "assistant":
                _render_assistant_extras(
                    sources=entry.get(
                        "sources",
                        [],
                    ),
                    chunks=entry.get(
                        "retrieved_chunks",
                        [],
                    ),
                    num_retrieved=(
                        len(
                            entry.get(
                                "retrieved_chunks",
                                [],
                            )
                        )
                        or None
                    ),
                )

    # ================================================================
    # CUSTOM CHATGPT-STYLE COMPOSER
    # ================================================================

    question = _render_custom_composer(
        pipeline
    )

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner(
            "Searching the astronomy knowledge base …"
        ):
            try:
                response = pipeline.answer_question(
                    question
                )
            except Exception as exc:
                response = {
                    "answer": (
                        "I couldn't process that question "
                        "right now."
                    ),
                    "sources": [],
                    "retrieved_chunks": [],
                    "error": str(exc),
                }

        answer = response.get(
            "answer",
            "No answer available.",
        )

        sources = response.get(
            "sources",
            [],
        )

        chunks = response.get(
            "retrieved_chunks",
            [],
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

            _render_assistant_extras(
                sources=sources,
                chunks=chunks,
                num_retrieved=response.get(
                    "num_retrieved"
                ),
            )

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
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