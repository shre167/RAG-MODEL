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

    -----------------------------------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

import html
from typing import Any


def inject_global_css() -> str:
    """Return CSS for the deep-space astronomy theme."""
    return """
<style>
@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
);

:root {
    --space-bg: #070b14;
    --space-bg-alt: #0c1220;
    --space-panel: rgba(14, 20, 36, 0.82);
    --space-panel-border: rgba(120, 150, 210, 0.18);

    --space-text: #e8edf7;
    --space-text-muted: #94a3b8;

    --space-accent: #6b9fd4;
    --space-accent-soft: rgba(107, 159, 212, 0.14);

    --space-cyan: #5ec4d4;
    --space-purple: #8b7ec8;

    --space-success: #4ade80;
    --space-warning: #fbbf24;
}

html,
body,
[class*="css"] {
    font-family:
        'Inter',
        -apple-system,
        BlinkMacSystemFont,
        'Segoe UI',
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            ellipse 120% 80% at 50% -20%,
            rgba(45, 70, 130, 0.22) 0%,
            transparent 55%
        ),
        radial-gradient(
            circle at 85% 15%,
            rgba(90, 60, 140, 0.12) 0%,
            transparent 35%
        ),
        radial-gradient(
            circle at 10% 80%,
            rgba(40, 90, 130, 0.10) 0%,
            transparent 40%
        ),
        linear-gradient(
            180deg,
            #070b14 0%,
            #0a0f1a 50%,
            #070b14 100%
        );

    color: var(--space-text);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;

    background-image:
        radial-gradient(
            1px 1px at 20px 30px,
            rgba(255,255,255,0.35),
            transparent
        ),
        radial-gradient(
            1px 1px at 80px 120px,
            rgba(255,255,255,0.25),
            transparent
        ),
        radial-gradient(
            1px 1px at 160px 60px,
            rgba(255,255,255,0.20),
            transparent
        ),
        radial-gradient(
            1px 1px at 240px 180px,
            rgba(255,255,255,0.30),
            transparent
        ),
        radial-gradient(
            1px 1px at 320px 40px,
            rgba(255,255,255,0.18),
            transparent
        ),
        radial-gradient(
            1px 1px at 400px 140px,
            rgba(255,255,255,0.22),
            transparent
        ),
        radial-gradient(
            1.5px 1.5px at 500px 90px,
            rgba(200,220,255,0.35),
            transparent
        ),
        radial-gradient(
            1px 1px at 600px 200px,
            rgba(255,255,255,0.20),
            transparent
        );

    background-size: 650px 250px;
    opacity: 0.55;

    animation: starDrift 120s linear infinite;
}

@keyframes starDrift {
    from {
        transform: translateY(0);
    }

    to {
        transform: translateY(-250px);
    }
}

.block-container {
    padding-top: 1.5rem;
    max-width: 920px;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(8, 12, 24, 0.97) 0%,
            rgba(10, 16, 30, 0.95) 100%
        );

    border-right:
        1px solid var(--space-panel-border);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem;
}

#MainMenu,
footer,
header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

/* =========================================================
   HERO
   ========================================================= */

.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;

    position: relative;
    z-index: 1;
}

.hero-stars {
    font-size: 0.85rem;
    letter-spacing: 0.6em;
    color: rgba(200, 220, 255, 0.35);
    margin-bottom: 1.5rem;
    user-select: none;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 300;
    letter-spacing: 0.35em;
    color: var(--space-text);

    margin:
        0
        0.25rem
        0
        0;

    text-transform: uppercase;
}

.hero-subtitle {
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.25em;
    color: var(--space-accent);

    margin:
        0
        0
        1.5rem
        0;

    text-transform: uppercase;
}

.hero-tagline {
    font-size: 1.05rem;
    color: var(--space-text-muted);
    margin-bottom: 2rem;
    font-weight: 300;
}

.hero-prompt-box {
    display: inline-block;

    padding:
        0.85rem
        2rem;

    border:
        1px solid
        var(--space-panel-border);

    border-radius: 8px;

    background:
        var(--space-panel);

    color:
        var(--space-text-muted);

    font-size: 0.92rem;

    margin-bottom: 2rem;
}

.hero-status-row {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    flex-wrap: wrap;
}

.hero-status-item {
    text-align: center;
}

.hero-status-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;

    color:
        var(--space-text-muted);

    margin-bottom: 0.35rem;
}

.hero-status-value {
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;

    color:
        var(--space-cyan);

    font-weight: 500;
}

.hero-status-value.inactive {
    color:
        var(--space-warning);
}

/* =========================================================
   SIDEBAR PANELS
   ========================================================= */

.panel-card {
    background:
        var(--space-panel);

    border:
        1px solid
        var(--space-panel-border);

    border-radius: 10px;

    padding:
        1rem
        1.1rem;

    margin-bottom: 0.85rem;
}

.panel-title {
    font-size: 0.68rem;
    letter-spacing: 0.16em;

    text-transform: uppercase;

    color:
        var(--space-accent);

    font-weight: 600;

    margin-bottom: 0.75rem;
}

.stat-grid {
    display: grid;

    grid-template-columns:
        1fr
        1fr;

    gap: 0.65rem;

    margin-bottom: 0.75rem;
}

.stat-item {
    background:
        rgba(255, 255, 255, 0.03);

    border-radius: 6px;

    padding:
        0.55rem
        0.65rem;

    text-align: center;
}

.stat-value {
    font-size: 1.35rem;
    font-weight: 600;

    color:
        var(--space-text);

    line-height: 1.2;
}

.stat-label {
    font-size: 0.62rem;
    letter-spacing: 0.08em;

    text-transform: uppercase;

    color:
        var(--space-text-muted);

    margin-top: 0.15rem;
}

.status-row {
    display: flex;

    justify-content:
        space-between;

    align-items: center;

    padding:
        0.35rem
        0;

    font-size: 0.82rem;

    color:
        var(--space-text-muted);
}

.status-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    display: inline-block;
}

.status-dot.active {
    background:
        var(--space-success);

    box-shadow:
        0 0 6px
        rgba(74, 222, 128, 0.5);
}

.status-dot.inactive {
    background: #64748b;
}

.status-dot.warning {
    background:
        var(--space-warning);

    box-shadow:
        0 0 6px
        rgba(251, 191, 36, 0.4);
}

.meta-text {
    font-size: 0.72rem;
    color:
        var(--space-text-muted);

    word-break: break-all;
}

/* =========================================================
   CHAT
   ========================================================= */

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;

    padding:
        0.35rem
        0 !important;
}

[data-testid="stChatMessage"]
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"]
[data-testid="stChatMessageAvatarAssistant"] {
    background:
        rgba(107, 159, 212, 0.15) !important;

    border:
        1px solid
        rgba(107, 159, 212, 0.25);
}

[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) [data-testid="stChatMessageContent"] {
    background:
        linear-gradient(
            135deg,
            rgba(50, 80, 140, 0.45) 0%,
            rgba(40, 60, 110, 0.35) 100%
        ) !important;

    border:
        1px solid
        rgba(107, 159, 212, 0.25);

    border-radius:
        14px
        14px
        4px
        14px !important;

    color:
        var(--space-text) !important;
}

[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] {
    background:
        var(--space-panel) !important;

    border:
        1px solid
        var(--space-panel-border);

    border-radius:
        4px
        14px
        14px
        14px !important;

    color:
        var(--space-text) !important;

    padding:
        0.85rem
        1rem !important;
}

/* =========================================================
   CUSTOM CHATGPT COMPOSER
   ========================================================= */

[data-testid="stHorizontalBlock"] {
    align-items: center;
}

/*
   Main text input.
   This is intentionally styled as the middle section
   of one unified composer.
*/

[data-testid="stTextInput"] {
    margin-bottom: 0 !important;
}

[data-testid="stTextInput"] > div {
    margin-bottom: 0 !important;
}

[data-testid="stTextInput"] input {
    min-height: 46px !important;

    background:
        rgba(14, 20, 36, 0.94) !important;

    border:
        1px solid
        rgba(120, 150, 210, 0.22) !important;

    border-radius:
        0 !important;

    color:
        var(--space-text) !important;

    font-size:
        0.94rem !important;

    padding:
        0.65rem
        0.85rem !important;

    box-shadow:
        none !important;

    transition:
        border-color 0.2s ease,
        background 0.2s ease !important;
}

[data-testid="stTextInput"] input:focus {
    border-color:
        rgba(107, 159, 212, 0.45) !important;

    background:
        rgba(16, 23, 41, 0.98) !important;

    box-shadow:
        none !important;
}

[data-testid="stTextInput"] input::placeholder {
    color:
        var(--space-text-muted) !important;
}

/*
   Remove Streamlit's visible label spacing.
*/

[data-testid="stTextInput"] label {
    display: none !important;
}

/*
   PLUS BUTTON
*/

[data-testid="stPopover"] {
    margin-bottom: 0 !important;
}

[data-testid="stPopover"] > button {
    min-height: 46px !important;
    min-width: 46px !important;

    padding:
        0.45rem
        0.7rem !important;

    background:
        rgba(14, 20, 36, 0.94) !important;

    border:
        1px solid
        rgba(120, 150, 210, 0.22) !important;

    border-right:
        none !important;

    border-radius:
        12px
        0
        0
        12px !important;

    color:
        var(--space-text) !important;

    font-size:
        1.25rem !important;

    transition:
        background 0.2s ease,
        color 0.2s ease !important;
}

[data-testid="stPopover"] > button:hover {
    background:
        rgba(107, 159, 212, 0.14) !important;

    color:
        #ffffff !important;
}

/*
   Send button
*/

[data-testid="stButton"] button {
    min-height: 46px !important;

    background:
        rgba(14, 20, 36, 0.94) !important;

    border:
        1px solid
        rgba(120, 150, 210, 0.22) !important;

    border-left:
        none !important;

    border-radius:
        0
        12px
        12px
        0 !important;

    color:
        var(--space-cyan) !important;

    font-size:
        1.05rem !important;

    transition:
        background 0.2s ease,
        color 0.2s ease,
        transform 0.15s ease !important;
}

[data-testid="stButton"] button:hover {
    background:
        rgba(107, 159, 212, 0.14) !important;

    color:
        #ffffff !important;
}

/*
   Popover body
*/

[data-testid="stPopoverBody"] {
    background:
        rgba(10, 16, 30, 0.98) !important;

    border:
        1px solid
        var(--space-panel-border) !important;

    border-radius:
        12px !important;

    box-shadow:
        0 18px 45px
        rgba(0, 0, 0, 0.45) !important;

    padding:
        1rem !important;
}

.add-file-title {
    color:
        var(--space-text);

    font-size:
        0.9rem;

    font-weight:
        600;

    margin-bottom:
        0.25rem;
}

.add-file-subtitle {
    color:
        var(--space-text-muted);

    font-size:
        0.72rem;

    line-height:
        1.5;

    margin-bottom:
        0.75rem;
}

.add-file-preserve {
    display:
        inline-flex;

    align-items:
        center;

    padding:
        0.32rem
        0.55rem;

    border-radius:
        6px;

    background:
        rgba(74, 222, 128, 0.07);

    border:
        1px solid
        rgba(74, 222, 128, 0.14);

    color:
        rgba(134, 239, 172, 0.86);

    font-size:
        0.66rem;

    margin-bottom:
        0.75rem;
}

.selected-file {
    display:
        flex;

    align-items:
        center;

    gap:
        0.45rem;

    margin-top:
        0.65rem;

    padding:
        0.55rem
        0.7rem;

    background:
        rgba(107, 159, 212, 0.07);

    border:
        1px solid
        rgba(107, 159, 212, 0.16);

    border-radius:
        7px;

    color:
        var(--space-text);

    font-size:
        0.74rem;

    word-break:
        break-all;
}

.selected-file-icon {
    color:
        var(--space-cyan);

    flex-shrink:
        0;
}

/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background:
        rgba(255, 255, 255, 0.02);

    border:
        1px dashed
        rgba(107, 159, 212, 0.30);

    border-radius:
        10px;

    padding:
        0.5rem;
}

[data-testid="stFileUploader"] section {
    background:
        transparent !important;
}

[data-testid="stFileUploader"] small {
    color:
        var(--space-text-muted) !important;
}

[data-testid="stFileUploader"] button {
    border-radius:
        7px !important;

    border:
        1px solid
        rgba(107, 159, 212, 0.25) !important;
}

/* =========================================================
   SIDEBAR BUTTONS
   ========================================================= */

[data-testid="stSidebar"] .stButton > button {
    width: 100%;

    background:
        rgba(107, 159, 212, 0.12);

    border:
        1px solid
        rgba(107, 159, 212, 0.25);

    color:
        var(--space-text);

    border-radius:
        8px;

    font-size:
        0.82rem;

    letter-spacing:
        0.04em;

    transition:
        background 0.2s,
        border-color 0.2s;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background:
        rgba(107, 159, 212, 0.22);

    border-color:
        rgba(107, 159, 212, 0.45);

    color:
        var(--space-text);
}

/* =========================================================
   SOURCES
   ========================================================= */

.sources-card {
    margin-top: 0.85rem;

    background:
        rgba(255, 255, 255, 0.02);

    border:
        1px solid
        var(--space-panel-border);

    border-radius:
        8px;

    padding:
        0.75rem
        1rem;
}

.sources-title {
    font-size:
        0.68rem;

    letter-spacing:
        0.14em;

    text-transform:
        uppercase;

    color:
        var(--space-accent);

    margin-bottom:
        0.5rem;

    font-weight:
        600;
}

.source-item {
    font-size:
        0.84rem;

    color:
        var(--space-text-muted);

    padding:
        0.2rem
        0;

    display:
        flex;

    align-items:
        center;

    gap:
        0.45rem;
}

.source-dot {
    color:
        var(--space-cyan);

    font-size:
        0.55rem;
}

/* =========================================================
   CONTEXT
   ========================================================= */

.context-chunk {
    background:
        rgba(255, 255, 255, 0.02);

    border-left:
        2px solid
        var(--space-accent);

    padding:
        0.65rem
        0.85rem;

    margin:
        0.5rem
        0;

    border-radius:
        0
        6px
        6px
        0;
}

.context-chunk-title {
    font-size:
        0.82rem;

    font-weight:
        600;

    color:
        var(--space-text);

    margin-bottom:
        0.35rem;
}

.context-chunk-text {
    font-size:
        0.82rem;

    color:
        var(--space-text-muted);

    line-height:
        1.5;
}

.context-chunk-score {
    font-size:
        0.68rem;

    color:
        rgba(148, 163, 184, 0.7);

    margin-top:
        0.3rem;
}

/* =========================================================
   BANNERS
   ========================================================= */

.dev-banner {
    background:
        rgba(251, 191, 36, 0.08);

    border:
        1px solid
        rgba(251, 191, 36, 0.25);

    border-radius:
        8px;

    padding:
        0.65rem
        1rem;

    font-size:
        0.82rem;

    color:
        #fcd34d;

    margin-bottom:
        1rem;
}

.warning-banner {
    background:
        rgba(251, 191, 36, 0.06);

    border:
        1px solid
        rgba(251, 191, 36, 0.2);

    border-radius:
        8px;

    padding:
        0.75rem
        1rem;

    color:
        #fcd34d;

    font-size:
        0.88rem;

    margin:
        1rem
        0;
}

/* =========================================================
   EXPANDER
   ========================================================= */

[data-testid="stExpander"] {
    background:
        transparent;

    border:
        1px solid
        var(--space-panel-border);

    border-radius:
        8px;
}

/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 768px) {
    .hero-title {
        font-size:
            1.6rem;

        letter-spacing:
            0.2em;
    }

    .hero-subtitle {
        font-size:
            0.8rem;
    }
}
</style>
"""


def render_landing_page(
    *,
    index_ready: bool,
    rag_ready: bool,
) -> str:
    kb_status = (
        "ACTIVE"
        if index_ready
        else "BUILD REQUIRED"
    )

    rag_status = (
        "ACTIVE"
        if rag_ready
        else "STANDBY"
    )

    kb_class = (
        ""
        if index_ready
        else " inactive"
    )

    rag_class = (
        ""
        if rag_ready
        else " inactive"
    )

    return f"""
<div class="hero-container">
    <div class="hero-stars">
        ✦ &nbsp; · &nbsp; ✧ &nbsp; · &nbsp; ✦
    </div>

    <h1 class="hero-title">
        Astronomy
    </h1>

    <p class="hero-subtitle">
        Knowledge Assistant
    </p>

    <p class="hero-tagline">
        Explore the universe through your intelligent knowledge base.
    </p>

    <div class="hero-prompt-box">
        Ask about stars, planets, missions, galaxies…
    </div>

    <div class="hero-status-row">

        <div class="hero-status-item">
            <div class="hero-status-label">
                Knowledge Base
            </div>

            <div class="hero-status-value{kb_class}">
                {kb_status}
            </div>
        </div>

        <div class="hero-status-item">
            <div class="hero-status-label">
                RAG Engine
            </div>

            <div class="hero-status-value{rag_class}">
                {rag_status}
            </div>
        </div>

    </div>
</div>
"""


def render_panel_card(
    title: str,
    inner_html: str,
) -> str:
    return f"""
<div class="panel-card">
    <div class="panel-title">
        {title}
    </div>

    {inner_html}
</div>
"""


def render_stat_grid(
    items: list[tuple[str, str]],
) -> str:
    cells = "".join(
        f"""
<div class="stat-item">
    <div class="stat-value">
        {value}
    </div>

    <div class="stat-label">
        {label}
    </div>
</div>
"""
        for value, label in items
    )

    return f"""
<div class="stat-grid">
    {cells}
</div>
"""


def render_status_row(
    label: str,
    active: bool,
    *,
    warning: bool = False,
) -> str:
    if warning:
        dot_class = "warning"
    elif active:
        dot_class = "active"
    else:
        dot_class = "inactive"

    return f"""
<div class="status-row">
    <span>
        {label}
    </span>

    <span class="status-dot {dot_class}"></span>
</div>
"""


def render_user_message(
    content: str,
) -> str:
    return f"""
<div class="chat-user-wrap">
    <div class="chat-user-bubble">
        {html.escape(content)}
    </div>
</div>
"""


def render_assistant_message(
    content: str,
) -> str:
    return f"""
<div class="chat-assistant-wrap">
    <div class="chat-assistant-bubble">
        {content}
    </div>
</div>
"""


def render_sources_card(
    sources: list[str],
    num_retrieved: int | None = None,
) -> str:
    if not sources:
        return ""

    items = "".join(
        f"""
<div class="source-item">
    <span class="source-dot">◉</span>
    {html.escape(src)}
</div>
"""
        for src in sources
    )

    meta = ""

    if num_retrieved is not None:
        meta = (
            '<div class="meta-text" '
            'style="margin-top:0.4rem;">'
            f"Retrieved chunks: {num_retrieved}"
            "</div>"
        )

    return f"""
<div class="sources-card">

    <div class="sources-title">
        📚 Sources
    </div>

    {items}

    {meta}

</div>
"""


def render_context_chunks(
    chunks: list[dict[str, Any]],
) -> str:
    parts: list[str] = []

    for chunk in chunks:
        heading = (
            chunk.get("section_heading")
            or ""
        )

        heading_html = (
            f" — <em>{html.escape(heading)}</em>"
            if heading
            else ""
        )

        score_val = chunk.get("score")
        score_html = ""

        if score_val is not None:
            score_html = (
                '<div class="context-chunk-score">'
                f"Relevance score: "
                f"{float(score_val):.4f}"
                "</div>"
            )

        parts.append(
            f"""
<div class="context-chunk">

    <div class="context-chunk-title">
        {html.escape(chunk.get("filename", ""))}
        {heading_html}
    </div>

    <div class="context-chunk-text">
        {html.escape(chunk.get("text", ""))}
    </div>

    {score_html}

</div>
"""
        )

    return "".join(parts)
    
