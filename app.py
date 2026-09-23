from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from src.config import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR
from src.document_loader import chunk_documents, list_txt_files, load_documents
from src.rag_pipeline import RAGPipeline
from src.rag_pipeline.observation_store import load_observations
from src.rag_pipeline.observatory import (
    build_observatory_snapshot,
    describe_evidence_level,
    export_trace_json,
    get_candidate_chunks,
    get_context_chunks,
    get_supporting_sources,
    has_trace,
    retrieval_mode_label,
)
from src.ui.astronomy_theme import (
    inject_global_css,
    render_arch_arrow,
    render_arch_legend,
    render_arch_node,
    render_arch_plane_open,
    render_architecture_node,
    render_chunk_map,
    render_citation_card,
    render_citation_coverage_badge,
    render_confidence_card,
    render_control_label,
    render_evaluation_criterion_card,
    render_evidence_badge,
    render_formula_card,
    render_html,
    render_kb_state_card,
    render_kv_rows,
    render_mode_chip,
    render_mode_description,
    render_nav_group_label,
    render_page_header,
    render_panel_card,
    render_pipeline_node,
    render_retrieval_health_card,
    render_section_title,
    render_sidebar_divider,
    render_simulation_tag,
    render_sources_card,
    render_stage_rail,
    render_stat_grid,
    render_status_card,
    render_status_row,
    render_trace_query_card,
)

# ============================================================================
# CONSTANTS
# ============================================================================

# These keys are the exact strings handed to the backend. They are not
# display strings and must not be renamed.
MODE_ORDER = ["vector", "bm25", "hybrid"]

MODES = {
    "vector": {
        "segment": "VECTOR",
        "short": "Vector",
        "caption": (
            "Dense semantic retrieval using the Chroma vector index."
        ),
    },
    "bm25": {
        "segment": "BM25",
        "short": "BM25",
        "caption": (
            "Lexical retrieval using the persistent BM25 index."
        ),
    },
    "hybrid": {
        "segment": "HYBRID",
        "short": "Hybrid",
        "caption": (
            "Dense + lexical retrieval fused using Reciprocal Rank Fusion."
        ),
    },
}

SEGMENT_TO_MODE = {MODES[key]["segment"]: key for key in MODE_ORDER}

PAGE_MISSION = "Mission Control"
PAGE_LAB = "Retrieval Lab"
PAGE_OBSERVATION = "Observation & Evaluation"
PAGE_DASHBOARD = "Evaluation Dashboard"
PAGE_BENCHMARK = "Benchmark"
PAGE_COMPARISON = "Query Comparison"
PAGE_INGEST = "Ingestion Pipeline"
PAGE_CHUNKS = "Chunk Monitor"
PAGE_ARCH = "System Architecture"

# (group label, [(page, icon)])
NAV_GROUPS = [
    ("Evaluation", [
        (PAGE_OBSERVATION, "Observation"),
        (PAGE_DASHBOARD, "Dashboard"),
        (PAGE_BENCHMARK, "Benchmark"),
        (PAGE_COMPARISON, "Compare"),
    ]),
    ("Mission Control", [(PAGE_MISSION, "🌌")]),
    ("Retrieval", [(PAGE_LAB, "🔬")]),
    ("Knowledge Base", [(PAGE_INGEST, "📥"), (PAGE_CHUNKS, "🧩")]),
    ("System", [(PAGE_ARCH, "🛰")]),
]

PAGES = [page for _, items in NAV_GROUPS for page, _ in items]


def _sidebar_nav_label(marker: str, page_name: str, icon: str) -> str:
    """Build sidebar button text; skip text icons that duplicate the page name."""
    if icon and not icon.isascii():
        return f"{marker}  {icon}  {page_name}"
    return f"{marker}  {page_name}"


INGESTION_STAGES = [
    ("📄", "Document", "Loaded"),
    ("🧹", "Text Cleaning", "Normalized"),
    ("✂️", "Chunking", "Segmented"),
    ("🧬", "Embedding", "Vectorized"),
    ("🗄️", "Chroma", "Stored"),
    ("🔎", "BM25", "Indexed"),
    ("✅", "Knowledge Ready", "Queryable"),
]

SIMULATION_STEPS = [
    ("01", "📄", "DOCUMENT DETECTED",
     "A TXT or Markdown file is discovered in the knowledge base "
     "directory and queued for processing."),
    ("02", "📖", "TEXT EXTRACTED",
     "Raw text is read and normalized. Whitespace is collapsed and "
     "section headings are detected so they can be kept as metadata."),
    ("03", "✂", "CONTENT CHUNKED",
     "The document is split into overlapping segments. Overlap keeps a "
     "sentence that crosses a boundary readable in both chunks."),
    ("04", "🧠", "EMBEDDING GENERATED",
     "Each chunk is converted into a numerical vector by the configured "
     "embedding model, capturing meaning rather than exact wording."),
    ("05", "🛰", "STORED IN CHROMA",
     "Vectors and metadata are written to the persistent Chroma "
     "collection, which survives restarts."),
    ("06", "🔎", "INDEXED IN BM25",
     "The same chunks are added to the persistent BM25 index for exact "
     "term matching, keeping rare names and identifiers findable."),
]

# Component descriptions for the architecture page. These describe the
# system as actually implemented — nothing here is aspirational.
ARCH_DESCRIPTIONS = [
    ("Document Loader",
     "Reads TXT and Markdown files from the knowledge base directory and "
     "detects their structure."),
    ("Text Processing",
     "Normalizes raw text and records section headings as metadata before "
     "the document is split."),
    ("Chunking",
     "Converts source documents into retrievable chunks used by both "
     "semantic and lexical retrieval."),
    ("Embedding Model",
     "Turns each chunk into a vector using the model named in the "
     "environment configuration."),
    ("Chroma Vector Store",
     "Persistent vector store used for dense semantic retrieval."),
    ("BM25 Index",
     "Lexical retrieval layer used for keyword and token-level matching. "
     "It requires no embedding call."),
    ("Query Preparation",
     "Normalizes the incoming question, expands aliases and produces the "
     "token list BM25 searches with."),
    ("Dense Search",
     "Embeds the prepared query and finds the nearest chunk vectors in "
     "Chroma."),
    ("BM25 Search",
     "Scores chunks by term overlap with the prepared query."),
    ("RRF Fusion",
     "Combines ranked dense and BM25 results using Reciprocal Rank "
     "Fusion. Used in Hybrid mode only. It is a ranking signal, not a "
     "confidence or probability."),
    ("Evidence Gate",
     "Evaluates whether the retrieved information is sufficient to "
     "answer, and decides whether the system should answer at all."),
    ("Context Selection",
     "Selects which of the surviving chunks are passed into generation, "
     "subject to chunk and character budgets."),
    ("Generation",
     "Uses the configured LLM to produce the final grounded response."),
    ("Answer",
     "The response returned to the user, together with its supporting "
     "sources and the runtime trace the Retrieval Lab reads."),
    ("Observation & Trace",
     "Captures full candidate journeys, index statistics, and retrieval states."),
    ("Retrieval Health Evaluation",
     "Evaluates 7 relative retrieval criteria at runtime, prioritizing direct "
     "answer support and relevance over agreement."),
    ("Claim Grounding & Citations",
     "Verifies factual claims against retrieved passages, mapping bracketed "
     "citations to chunk sources and flagging unsupported claims."),
    ("Evidence Confidence",
     "Computes evidence-backed confidence (HIGH/MEDIUM/LOW) with explainable reasons."),
    ("Knowledge Base State",
     "Tracks versioning, document counts, and Chroma/BM25 index synchronization."),
]

RETRIEVAL_DIAGRAM = """\
                    USER QUERY
                        │
                        ▼
                Query Preparation
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Dense Retrieval           BM25
             │                     │
             └──────────┬──────────┘
                        ▼
                       RRF
                        │
                        ▼
                Evidence Evaluation
                        │
                        ▼
                 Context Selection
                        │
                        ▼
                  LLM Generation
                        │
                        ▼
                      Answer"""

INGESTION_DIAGRAM = """\
Documents
    │
    ▼
Cleaning
    │
    ▼
Chunking
    │
    ├───────────────┐
    ▼               ▼
Embeddings        BM25
    │
    ▼
Chroma"""


# ============================================================================
# STREAMLIT COMPATIBILITY
# ============================================================================

def _layout_kwargs() -> dict:
    """
    Full-width keyword arguments for the installed Streamlit version.

    Streamlit 1.49 replaced use_container_width=True with width="stretch"
    and deprecated the old name. Resolving this at runtime keeps the same
    source file working across both machines without edits.
    """
    try:
        parts = st.__version__.split(".")
        version = (int(parts[0]), int(parts[1]))
    except Exception:  # noqa: BLE001
        return {"use_container_width": True}

    if version >= (1, 49):
        return {"width": "stretch"}

    return {"use_container_width": True}


WIDE = _layout_kwargs()


def _dataframe(rows) -> None:
    """Render a table full width, without version-specific warnings."""
    st.dataframe(rows, hide_index=True, **WIDE)


def _segmented_control(options: list, default: str, key: str) -> str:
    """
    Render a real segmented control.

    Uses st.segmented_control where available (Streamlit 1.40+), then
    st.pills, and finally falls back to a horizontal radio which the
    stylesheet strips of its radio circles. In every case the returned
    value is one of `options`.
    """
    if hasattr(st, "segmented_control"):
        try:
            value = st.segmented_control(
                "Retrieval engine",
                options,
                default=default,
                key=key,
                label_visibility="collapsed",
                **WIDE,
            )
        except TypeError:
            # Older Streamlit accepts segmented_control but not the
            # width/use_container_width kwarg — retry without it rather
            # than losing the control entirely.
            try:
                value = st.segmented_control(
                    "Retrieval engine",
                    options,
                    default=default,
                    key=key,
                    label_visibility="collapsed",
                )
            except TypeError:
                value = None

        return value if value in options else default

    if hasattr(st, "pills"):
        try:
            value = st.pills(
                "Retrieval engine",
                options,
                default=default,
                key=key,
                label_visibility="collapsed",
            )
            return value if value in options else default
        except TypeError:
            pass

    value = st.radio(
        "Retrieval engine",
        options,
        index=options.index(default),
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )

    return value if value in options else default


# ============================================================================
# BACKEND ACCESS HELPERS
# ============================================================================

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


def _pipeline_status(pipeline: RAGPipeline) -> dict:
    """Read pipeline.status() without letting a backend error kill the UI."""
    try:
        status = pipeline.status()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user below
        st.session_state.status_error = str(exc)
        return {}

    st.session_state.pop("status_error", None)

    return status or {}


def _collection_count(pipeline: RAGPipeline):
    """Actual Chroma collection count, or None when it cannot be read."""
    try:
        return int(pipeline.vector_store.get_collection_count())
    except Exception:  # noqa: BLE001
        return None


def _bm25_indexed_count(status: dict):
    """
    Number of chunks in the BM25 index, if the backend reports it.

    Returns None rather than a guess when status exposes no value.
    """
    for key in (
        "bm25_chunks",
        "bm25_documents",
        "bm25_count",
        "bm25_indexed_chunks",
        "bm25_size",
    ):
        value = status.get(key)

        if isinstance(value, int) and not isinstance(value, bool):
            return value

    return None


def _llm_model(status: dict):
    """
    Configured LLM model name.

    Read from pipeline status first, then from the environment, using the
    existing variable names. Never hardcoded.
    """
    for key in ("llm_model", "generation_model", "model"):
        value = status.get(key)

        if value:
            return str(value)

    for env_name in ("LLM_MODEL", "GENERATION_MODEL"):
        value = os.getenv(env_name)

        if value:
            return value

    return None


def _chunk_config(pipeline: RAGPipeline) -> tuple:
    """Configured chunk size and overlap, read from the live pipeline."""
    return (
        getattr(pipeline, "chunk_size", None),
        getattr(pipeline, "chunk_overlap", None),
    )


def _kb_signature(kb_path: Path) -> tuple:
    """
    Fingerprint of the knowledge base directory.

    Used as a cache key so chunk previews refresh when files change.
    """
    signature = []

    try:
        for path in sorted(list_txt_files(kb_path)):
            stat = path.stat()
            signature.append((path.name, stat.st_size, int(stat.st_mtime)))
    except Exception:  # noqa: BLE001
        return tuple()

    return tuple(signature)


@st.cache_data(show_spinner=False)
def _compute_chunks(
    kb_path: str,
    chunk_size: int,
    chunk_overlap: int,
    signature: tuple,
) -> list:
    """
    Recompute chunks for inspection using the backend's own chunker.

    This is the same chunking function the ingestion path uses, so the
    preview reflects real behaviour. It does not write to Chroma or BM25.
    """
    documents = load_documents(Path(kb_path))

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return [c for c in chunks if (c.get("text") or "").strip()]


def _load_chunks(pipeline: RAGPipeline) -> list:
    """Cached chunk preview for the knowledge base."""
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    size, overlap = _chunk_config(pipeline)

    if size is None or overlap is None:
        return []

    try:
        return _compute_chunks(
            str(kb_path),
            int(size),
            int(overlap),
            _kb_signature(kb_path),
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Chunk preview unavailable: {exc}")
        return []


def _document_characters(path: Path):
    """Character count of a knowledge base file."""
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return None


def _na(value):
    """Map empty values to None so the UI renders a muted N/A."""
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    return value


def _service_error(title: str, detail: str = "") -> None:
    """
    Show a professional error panel instead of a raw stack trace.

    The underlying exception text is kept behind an expander so it is
    available for debugging without being pushed at a normal user.
    """
    st.error(
        f"⚠ {title}\n\n"
        "The configured retrieval, embedding or generation service could "
        "not be reached. Check the environment and API configuration."
    )

    if detail:
        with st.expander("Technical detail"):
            st.code(str(detail), language="text")


# ============================================================================
# RESPONSE / TRACE HELPERS
# ============================================================================

def _build_meta(response: dict, requested_mode: str) -> dict:
    """
    Build display metadata for one answer.

    Everything here is read from the runtime trace. Nothing is synthesized.
    """
    trace = response.get("trace") or {}

    snapshot = build_observatory_snapshot(trace) if trace else {}

    evidence = snapshot.get("evidence", {}) if snapshot else {}
    retrieval = snapshot.get("retrieval", {}) if snapshot else {}

    return {
        "requested_mode": requested_mode,
        "mode_label": retrieval.get(
            "mode_label",
            retrieval_mode_label(requested_mode),
        ),
        "evidence_level": evidence.get("level"),
        "evidence_label": describe_evidence_level(evidence.get("level")),
        "supporting_sources": get_supporting_sources(trace, response),
        "context_chunks": get_context_chunks(trace),
        "candidates": get_candidate_chunks(trace, response),
        "candidate_count": retrieval.get("candidate_count"),
        "trace_available": bool(trace),
        "error": response.get("error", ""),
        "evaluation": response.get("evaluation") or snapshot.get("evaluation", {}),
        "citations": response.get("citations") or snapshot.get("citations", []),
        "citation_coverage": response.get("citation_coverage") or snapshot.get("citation_coverage", {}),
        "confidence": response.get("confidence") or snapshot.get("confidence", {}),
        "kb_state": response.get("kb_state") or snapshot.get("kb_state", {}),
        "candidate_journey": trace.get("candidate_journey") or snapshot.get("candidate_journey", []),
    }


def _render_chunk_list(chunks: list, mode_label: str = "") -> None:
    """Render chunks with native components so text formatting survives."""
    for index, chunk in enumerate(chunks, start=1):
        filename = chunk.get("filename") or "unknown source"
        chunk_id = chunk.get("chunk_id") or "unknown id"

        parts = [f"**{index}. {filename}**", f"`{chunk_id}`"]

        if mode_label:
            parts.append(mode_label)

        score = chunk.get("score")

        if score is not None:
            parts.append(f"{chunk.get('score_label', 'Score')}: {score}")

        st.markdown(" · ".join(parts))

        text = chunk.get("text") or ""

        if text:
            st.code(text, language="text")
        else:
            st.caption("Chunk text is not present in the trace.")


def _render_answer_extras(meta: dict) -> None:
    """Render mode, evidence, confidence, citations, sources and retrieved context under an answer."""
    chip_col, evidence_col = st.columns([1, 1])

    with chip_col:
        render_html(render_mode_chip(meta.get("mode_label", "")))

    with evidence_col:
        if meta.get("trace_available"):
            render_html(
                render_evidence_badge(
                    meta.get("evidence_label", "Not available"),
                    str(meta.get("evidence_level") or "none"),
                )
            )

    confidence = meta.get("confidence")
    if confidence:
        render_html(render_confidence_card(confidence))

    coverage = meta.get("citation_coverage")
    kb_state = meta.get("kb_state")
    if coverage or kb_state:
        sub_col1, sub_col2 = st.columns([1, 1])
        with sub_col1:
            if coverage:
                render_html(render_citation_coverage_badge(coverage))
        with sub_col2:
            if kb_state:
                v = kb_state.get("version", 1)
                sync = "✓ Synced" if kb_state.get("indexes_consistent", True) else "⚠ Mismatch"
                st.caption(f"Knowledge Base: **v{v}** ({sync})")

    sources = meta.get("supporting_sources") or []

    if sources:
        render_html(render_sources_card(sources))
    else:
        st.caption("No supporting sources were selected for this answer.")

    citations = meta.get("citations") or []
    if citations:
        with st.expander(f"Claim Citations & Evidence Passages ({len(citations)})", expanded=False):
            for c in citations:
                render_html(render_citation_card(c))

    context_chunks = meta.get("context_chunks") or []
    candidates = meta.get("candidates") or []

    if context_chunks:
        label = (
            f"Supporting context — {len(context_chunks)} chunks "
            "sent to the model"
        )

        with st.expander(label):
            _render_chunk_list(
                context_chunks,
                mode_label=meta.get("mode_label", ""),
            )

    elif candidates:
        with st.expander(
            f"Retrieved candidates — {len(candidates)} chunks"
        ):
            st.caption(
                "The trace did not record the final context bodies. These "
                "are retrieval candidates, which are not the same as the "
                "chunks that supported the answer."
            )
            _render_chunk_list(
                candidates,
                mode_label=meta.get("mode_label", ""),
            )

    if meta.get("error"):
        _service_error("GENERATION SERVICE UNAVAILABLE", meta["error"])


# ============================================================================
# NOTICES
# ============================================================================

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


# ============================================================================
# INGESTION
# ============================================================================

def _ingest_uploaded_file(pipeline: RAGPipeline, uploaded, rerun: bool = False):
    """
    Run the existing incremental ingestion path for an uploaded file.

    This calls pipeline.ingest_file(...) unchanged. Existing chunks are
    never removed by this operation.
    """
    try:
        content = uploaded.getvalue().decode("utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read {uploaded.name}: {exc}")
        return None

    with st.spinner(f"Ingesting {uploaded.name} …"):
        try:
            result = pipeline.ingest_file(
                file_path=uploaded.name,
                content=content,
            )
        except Exception as exc:  # noqa: BLE001
            _service_error("INGESTION SERVICE UNAVAILABLE", exc)
            st.caption("No existing data was removed.")
            return None

    result = dict(result or {})
    result["_filename"] = uploaded.name
    result["_characters"] = len(content)

    st.session_state.last_ingestion = result

    if result.get("skipped"):
        message = (
            f"{uploaded.name} is already up-to-date. "
            "No embeddings were changed."
        )
        notice_type = "info"
    else:
        message = (
            f"{uploaded.name} added — "
            f"{result.get('chunks_added', 'unknown')} new chunks indexed. "
            f"Total indexed: {result.get('collection_count', 'unknown')}."
        )
        notice_type = "success"

    st.session_state.ingestion_notice = {
        "type": notice_type,
        "message": message,
    }

    if rerun:
        st.rerun()

    return result


def _replay_ingestion_stages(result: dict) -> None:
    """
    Replay the ingestion stages as a visualization.

    This animation is drawn after ingestion has already finished. It is
    not a live event stream. Stage details show real values from the
    ingestion result, and are blank where the backend reports nothing.
    """
    details = [
        result.get("_filename"),
        None,
        result.get("chunks_created", result.get("chunks")),
        None,
        result.get("collection_count"),
        result.get("bm25_chunks", result.get("bm25_synced")),
        "Queryable",
    ]

    placeholder = st.empty()
    progress = st.progress(0.0)

    total = len(INGESTION_STAGES)

    for step in range(1, total + 1):
        stages = []

        for index, stage in enumerate(INGESTION_STAGES):
            state = "done" if index < step else "pending"
            detail = details[index]

            stages.append(
                (state, stage[1], detail if detail is not None else "")
            )

        with placeholder.container():
            render_html(render_stage_rail(stages))

        progress.progress(step / total)
        time.sleep(0.12)

    progress.empty()


def _render_ingestion_result(result: dict) -> None:
    """Show the real values returned by pipeline.ingest_file(...)."""
    render_html(render_section_title("Ingestion result"))

    skipped = bool(result.get("skipped"))

    if skipped:
        st.info(
            "This file was unchanged, so incremental ingestion skipped it. "
            "No embeddings were recomputed."
        )

    col1, col2, col3 = st.columns(3)

    col1.metric("Characters", result.get("_characters", "N/A"))

    chunks_created = result.get("chunks_created", result.get("chunks"))
    col2.metric(
        "Chunks created",
        chunks_created if chunks_created is not None else "N/A",
    )

    chunks_added = result.get("chunks_added")
    col3.metric(
        "Chunks added",
        chunks_added if chunks_added is not None else "N/A",
    )

    render_html(
        render_kv_rows(
            [
                ("File", _na(result.get("_filename"))),
                ("Skipped (unchanged)", "Yes" if skipped else "No"),
                ("Collection count", _na(result.get("collection_count"))),
                (
                    "BM25 synchronization",
                    _na(
                        result.get(
                            "bm25_synced",
                            result.get("bm25_status"),
                        )
                    ),
                ),
            ]
        )
    )

    missing = [
        key
        for key in ("chunks_created", "bm25_synced")
        if key not in result
    ]

    if missing:
        st.caption(
            "Fields shown as N/A are not reported by the current ingestion "
            "backend. They are not estimated here."
        )


# ============================================================================
# PAGE — MISSION CONTROL
# ============================================================================

def _render_overview_cards(
    status: dict,
    collection_count,
    llm_model,
) -> None:
    """Four status cards driven entirely by real application state."""
    col1, col2, col3, col4 = st.columns(4)

    index_ready = bool(status.get("index_available"))
    bm25_ready = bool(status.get("bm25_available"))
    embedding_model = status.get("embedding_model")

    with col1:
        render_html(
            render_status_card(
                "Knowledge Core",
                collection_count,
                "Indexed chunks",
            )
        )

    with col2:
        render_html(
            render_status_card(
                "Vector Index",
                "ONLINE" if index_ready else "OFFLINE",
                "Chroma",
                state="online" if index_ready else "offline",
                small=True,
            )
        )

    with col3:
        render_html(
            render_status_card(
                "BM25 Index",
                "ONLINE" if bm25_ready else "OFFLINE",
                "Lexical search",
                state="online" if bm25_ready else "offline",
                small=True,
            )
        )

    with col4:
        render_html(
            render_status_card(
                "Language Model",
                llm_model if llm_model else None,
                "Configured model",
                state="online" if llm_model else "unknown",
                small=True,
            )
        )

    st.caption(
        f"Embedding model: {embedding_model or 'N/A'} · "
        f"Source documents: {status.get('txt_files_found', 'N/A')}"
    )


def _render_mission_control(
    pipeline: RAGPipeline,
    status: dict,
    collection_count,
) -> None:
    render_html(
        render_page_header(
            "ASTRONOMY OBSERVATORY",
            "Mission Control",
            "Knowledge Intelligence & Retrieval System",
        )
    )

    _render_overview_cards(status, collection_count, _llm_model(status))

    st.write("")

    render_html(render_section_title("🌌 Astronomy Knowledge Assistant"))

    st.caption(
        "Ask questions about planets, missions, cosmology, telescopes, "
        "astrophysics and space technology."
    )

    # ----------------------------------------------------------------
    # RETRIEVAL ENGINE — segmented control wired to the real backend
    # ----------------------------------------------------------------
    current = st.session_state.retrieval_mode

    if current not in MODE_ORDER:
        current = "hybrid"

    render_html(render_control_label("Retrieval Engine"))

    segment = _segmented_control(
        [MODES[key]["segment"] for key in MODE_ORDER],
        MODES[current]["segment"],
        key="retrieval_segment",
    )

    selected_mode = SEGMENT_TO_MODE.get(segment, current)
    st.session_state.retrieval_mode = selected_mode

    render_html(render_mode_description(MODES[selected_mode]["caption"]))

    if selected_mode in ("vector", "hybrid") and not status.get(
        "index_available"
    ):
        st.warning(
            "The vector index is not available. BM25 mode remains usable "
            "if the lexical index is present."
        )

    if selected_mode in ("bm25", "hybrid") and not status.get(
        "bm25_available"
    ):
        st.warning("The BM25 index is not available for this query.")

    # ----------------------------------------------------------------
    # HISTORY
    # ----------------------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not st.session_state.chat_history:
        st.info(
            "No queries have been run yet. Ask a question below to bring "
            "the retrieval pipeline online."
        )

        if not status.get("index_available"):
            st.warning(
                "No vector index found. Use Sync Knowledge Base in the "
                "sidebar, or run: python ingest.py"
            )

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

            if entry["role"] == "assistant":
                _render_answer_extras(entry.get("meta", {}))

    # ----------------------------------------------------------------
    # NEW QUESTION
    # ----------------------------------------------------------------
    question = _render_custom_composer(pipeline)

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner(
        f"Searching with {MODES[selected_mode]['short']} retrieval …"
    ):
        try:
            response = pipeline.answer_question(
                question,
                retrieval_mode=selected_mode,
            )
        except Exception as exc:  # noqa: BLE001
            response = {
                "answer": (
                    "I could not complete that query. If the embedding or "
                    "generation service is unavailable, BM25 mode does not "
                    "require embeddings and may still work."
                ),
                "sources": [],
                "retrieved_chunks": [],
                "error": str(exc),
                "retrieval_mode": selected_mode,
            }

    st.session_state.last_response = response

    answer = response.get("answer", "No answer available.")
    meta = _build_meta(response, selected_mode)

    with st.chat_message("assistant"):
        st.markdown(answer)
        _render_answer_extras(meta)

    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "meta": meta}
    )

    st.rerun()


# ============================================================================
# COMPOSER
# ============================================================================

def _render_custom_composer(pipeline: RAGPipeline):
    """
    Render the astronomy composer.

    The question input and send button are inside a Streamlit form, so
    pressing Enter submits the question just like clicking the arrow.
    A versioned widget key creates a fresh empty input after submission.
    """
    if "composer_version" not in st.session_state:
        st.session_state.composer_version = 0

    plus_col, composer_col = st.columns([0.75, 10.25], gap="small")

    # ----------------------------------------------------------------
    # PLUS / KNOWLEDGE UPLOAD
    # ----------------------------------------------------------------
    with plus_col:
        with st.popover("＋", help="Add knowledge to the knowledge base"):
            st.markdown("**Add knowledge**")
            st.caption(
                "Upload a TXT or Markdown file. Indexing is incremental, "
                "so existing knowledge is preserved."
            )

            uploaded = st.file_uploader(
                "Choose a file",
                type=["txt", "md"],
                key="composer_file_uploader",
                label_visibility="collapsed",
            )

            if uploaded is not None:
                st.caption(f"◈ {uploaded.name}")

                if st.button(
                    "Add to Knowledge Base",
                    key="composer_add_file",
                    **WIDE,
                ):
                    _ingest_uploaded_file(pipeline, uploaded, rerun=True)

    # ----------------------------------------------------------------
    # QUESTION COMPOSER
    # ----------------------------------------------------------------
    send_clicked = False
    question = ""

    with composer_col:
        with st.form(
            key=f"question_form_{st.session_state.composer_version}",
            clear_on_submit=False,
            border=False,
        ):
            input_col, send_col = st.columns([9.35, 0.65], gap="small")

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
                    help="Send question",
                    **WIDE,
                )

    if send_clicked:
        question = (question or "").strip()

        if not question:
            return None

        # Do not assign directly to the current widget's session-state key.
        # Instead, create a new widget key on the next rerun. This clears
        # the input without triggering Streamlit's widget-state exception.
        st.session_state.composer_version += 1

        return question

    return None


# ============================================================================
# PAGE — RETRIEVAL LAB
# ============================================================================

def _render_retrieval_pipeline_flow(retrieval: dict) -> None:
    """
    Draw the retrieval flow for the mode that actually ran.

    Vector mode does not show BM25, BM25 mode does not show dense search,
    and RRF only appears when fusion actually happened.
    """
    dense = bool(retrieval.get("dense_enabled"))
    bm25 = bool(retrieval.get("bm25_enabled"))
    rrf = bool(retrieval.get("rrf_enabled"))

    render_html(render_section_title("Retrieval flow for this query"))

    top = st.columns(3)

    with top[1]:
        render_html(
            render_arch_node("❓", "Query", "user input", "processing")
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "⚙", "Query Preparation", "normalize / tokenize", "processing"
            )
        )
        render_html(render_arch_arrow())

    # Retrieval branch — only the retrievers that ran.
    if dense and bm25:
        branch = st.columns(2)

        with branch[0]:
            render_html(
                render_arch_node(
                    "◈", "Dense Search", "Chroma vectors", "retrieval"
                )
            )

        with branch[1]:
            render_html(
                render_arch_node(
                    "⌕", "BM25 Search", "lexical match", "retrieval"
                )
            )

    else:
        single = st.columns(3)

        with single[1]:
            if dense:
                render_html(
                    render_arch_node(
                        "◈", "Dense Search", "Chroma vectors", "retrieval"
                    )
                )
            elif bm25:
                render_html(
                    render_arch_node(
                        "⌕", "BM25 Search", "lexical match", "retrieval"
                    )
                )
            else:
                render_html(
                    render_arch_node(
                        "∅", "No retriever recorded", "", "retrieval"
                    )
                )

    tail = st.columns(3)

    with tail[1]:
        render_html(render_arch_arrow())

        if rrf:
            render_html(
                render_arch_node(
                    "◉",
                    "RRF Fusion",
                    f"k = {retrieval.get('rrf_k', 'N/A')}",
                    "retrieval",
                )
            )
        else:
            render_html(
                render_arch_node(
                    "◌", "RRF — Not used", "single retriever", "retrieval"
                )
            )

        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "⚖", "Evidence Gate", "sufficiency check", "decision"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "🗂", "Context Selection", "chunk budget", "processing"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "✦", "Generation", "configured LLM", "generation"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "◆", "Answer", "sources + trace", "generation"
            )
        )


def _render_retrieval_lab(response) -> None:
    render_html(
        render_page_header(
            "RUNTIME DIAGNOSTICS",
            "Retrieval Lab",
            "Every panel below reads the trace of the most recent query. "
            "Opening this page runs no retrieval and calls no model.",
        )
    )

    if not response:
        st.info(
            "🔬 No retrieval trace available.\n\n"
            "Run a query from Mission Control to inspect the retrieval "
            "pipeline here."
        )
        return

    if not has_trace(response):
        st.warning(
            "The last response carried no runtime trace, so there is "
            "nothing to observe. This usually means the query failed "
            "before retrieval ran."
        )

        if response.get("error"):
            _service_error("RETRIEVAL SERVICE UNAVAILABLE", response["error"])

        return

    trace = response.get("trace") or {}
    snapshot = build_observatory_snapshot(trace)

    retrieval = snapshot.get("retrieval", {})

    _lab_section_query(snapshot.get("query", {}))
    _lab_section_mode(retrieval)
    _render_retrieval_pipeline_flow(retrieval)

    if retrieval.get("dense_enabled"):
        _lab_section_dense(retrieval)

    if retrieval.get("bm25_enabled"):
        _lab_section_bm25(retrieval)

    if retrieval.get("rrf_enabled"):
        _lab_section_rrf(retrieval)
    else:
        render_html(render_section_title("Reciprocal Rank Fusion"))
        st.info(
            "RRF — Not used. Fusion applies to Hybrid mode only, where "
            "two ranked lists exist to combine."
        )

    _lab_section_evidence(snapshot.get("evidence", {}))
    _lab_section_context(trace, snapshot.get("context", {}))
    _lab_section_generation(snapshot.get("generation", {}))
    _lab_section_grounding_chain(snapshot.get("query", {}), snapshot.get("grounding", {}))

    with st.expander("Raw trace (sanitized)"):
        st.caption(
            "Secret-looking fields are stripped by "
            "observatory.export_trace_json()."
        )
        st.json(export_trace_json(trace))


def _lab_section_grounding_chain(query: dict, grounding: dict) -> None:
    """Render the auditable Query -> Evidence -> Claim -> Citation chain."""
    render_html(render_section_title("Grounding chain"))
    st.caption("Query → selected evidence passage → generated claim → citation verification")
    rows = []
    for item in grounding.get("claim_to_citation", []):
        evidence = item.get("evidence", {})
        rows.append({
            "Query": query.get("original", ""),
            "Evidence": f"{evidence.get('filename', '')} / {evidence.get('chunk_id', '')}",
            "Claim": item.get("claim", ""),
            "Citation": item.get("citation", ""),
            "Verdict": item.get("status", ""),
        })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No generated claims were available for post-generation grounding.")


def _render_observation_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Observation & Evaluation", "Persistent, local records of query execution and explainable evaluation."))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet. Run a query from Mission Control.")
        return
    choices = {f"{row['timestamp']} · {row['raw_query']}": row for row in rows}
    selected = choices[st.selectbox("Recorded query", list(choices))]
    trace = selected.get("trace", {})
    snapshot = build_observatory_snapshot(trace)
    st.json({"query_id": selected["query_id"], "normalized_query": selected["normalized_query"], "latency_ms": selected["latency_ms"], "embedding_model": selected["embedding_model"]})
    st.subheader("Query → Dense/BM25/RRF → Evidence → Context → Answer")
    st.json({"dense": snapshot.get("retrieval", {}).get("raw_dense", []), "bm25": snapshot.get("retrieval", {}).get("raw_bm25", []), "rrf": snapshot.get("retrieval", {}).get("rrf_calculation", []), "selected_evidence": trace.get("selected_evidence", []), "context": trace.get("context", {}), "grounding": trace.get("grounding", {})})
    st.subheader("Evaluation and insights")
    st.json({"retrieval": selected.get("evaluation", {}), "answer": selected.get("answer_evaluation", {})})


def _render_dashboard_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Evaluation Dashboard", "Aggregate live heuristics from recorded observations; these are not accuracy metrics."))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet.")
        return
    table = [{"query": row["raw_query"], "latency_ms": row["latency_ms"], "retrieval_health": row.get("evaluation", {}).get("retrieval_health_score"), "citation_coverage": row.get("answer_evaluation", {}).get("citation_coverage", {}).get("score")} for row in rows]
    st.dataframe(table, use_container_width=True, hide_index=True)


def _render_benchmark_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("EVALUATION", "Benchmark", "Labelled queries are evaluated separately from live heuristic observations."))
    st.info("Add labelled cases with query, relevant_documents/relevant_chunks, expected_facets, and optional reference_answer. Benchmark calculation is available in src/rag_pipeline/benchmark.py.")


def _render_comparison_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Query Comparison", "Compare two recorded executions without rerunning retrieval."))
    rows = load_observations(pipeline.vectorstore_path)
    if len(rows) < 2:
        st.info("Run at least two queries to compare observations.")
        return
    labels = [f"{row['timestamp']} · {row['raw_query']}" for row in rows]
    left, right = st.columns(2)
    with left:
        st.json(rows[labels.index(st.selectbox("First query", labels, key="compare_a"))])
    with right:
        st.json(rows[labels.index(st.selectbox("Second query", labels, index=1, key="compare_b"))])


# ============================================================================
# PAGE — INGESTION PIPELINE
# ============================================================================

def _render_ingestion_architecture() -> None:
    """Visual ingestion architecture using themed nodes."""
    top = st.columns(3)

    with top[1]:
        render_html(
            render_arch_node("📄", "Document", "TXT / MD", "storage")
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "📥", "Document Loader", "read + detect", "processing"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "🧹", "Text Processing", "normalize", "processing"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "✂", "Chunking", "size + overlap", "processing"
            )
        )
        render_html(render_arch_arrow("╱  ╲"))

    branch = st.columns(2)

    with branch[0]:
        render_html(
            render_arch_node(
                "🧬", "Embedding Model", "vectorize", "processing"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "🗄", "Chroma Vector Store", "dense index", "storage"
            )
        )

    with branch[1]:
        render_html(
            render_arch_node(
                "🔎", "BM25", "term weighting", "retrieval"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "📚", "BM25 Index", "lexical index", "storage"
            )
        )


def _render_ingestion_simulation(pipeline: RAGPipeline) -> None:
    """
    Educational, read-only walkthrough of the ingestion stages.

    This section never calls ingest_file, ingest_documents, or any write
    path. It only reads configuration and the already-computed chunk
    preview, so it cannot modify Chroma, BM25 or the knowledge base.
    """
    st.warning(
        "EDUCATIONAL INGESTION SIMULATION — ILLUSTRATIVE ONLY. "
        "This walkthrough does not modify Chroma, BM25, the vectorstore "
        "or the knowledge base."
    )

    render_html(render_simulation_tag("Simulation — no database writes"))

    size, overlap = _chunk_config(pipeline)
    chunks = _load_chunks(pipeline)

    if st.button("▶ Run simulation", key="sim_run"):
        placeholder = st.empty()
        progress = st.progress(0.0)

        for step in range(1, len(SIMULATION_STEPS) + 1):
            stages = []

            for index, item in enumerate(SIMULATION_STEPS):
                state = "done" if index < step else "pending"
                stages.append((state, f"{item[0]}  {item[2]}", ""))

            with placeholder.container():
                render_html(render_stage_rail(stages))

            progress.progress(step / len(SIMULATION_STEPS))
            time.sleep(0.25)

        progress.empty()
        st.success("Simulation complete. No data was written.")

    for number, icon, title, description in SIMULATION_STEPS:
        with st.expander(f"{number}  {icon}  {title}"):
            st.write(description)

            if title == "CONTENT CHUNKED":
                render_html(
                    render_kv_rows(
                        [
                            ("Configured chunk size", _na(size)),
                            ("Configured overlap", _na(overlap)),
                        ]
                    )
                )

                if chunks:
                    st.caption(
                        "A real chunk produced by the backend's chunker:"
                    )
                    st.code(
                        (chunks[0].get("text") or "")[:400],
                        language="text",
                    )


def _render_ingestion_page(
    pipeline: RAGPipeline,
    status: dict,
    collection_count,
) -> None:
    render_html(
        render_page_header(
            "KNOWLEDGE INGESTION",
            "Ingestion Pipeline",
            "How documents become searchable knowledge.",
        )
    )

    # ----------------------------------------------------------------
    # REAL KNOWLEDGE CORE STATUS
    # ----------------------------------------------------------------
    render_html(render_section_title("Knowledge core"))

    size, overlap = _chunk_config(pipeline)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_html(
            render_status_card(
                "Documents",
                status.get("txt_files_found"),
                "Source files",
            )
        )

    with col2:
        render_html(
            render_status_card(
                "Chunks",
                collection_count,
                "Indexed",
            )
        )

    with col3:
        ready = bool(status.get("index_available"))
        render_html(
            render_status_card(
                "Vector Index",
                "ONLINE" if ready else "OFFLINE",
                "Chroma",
                state="online" if ready else "offline",
                small=True,
            )
        )

    with col4:
        ready = bool(status.get("bm25_available"))
        render_html(
            render_status_card(
                "BM25 Index",
                "ONLINE" if ready else "OFFLINE",
                "Lexical",
                state="online" if ready else "offline",
                small=True,
            )
        )

    render_html(
        render_kv_rows(
            [
                ("Chunk size", _na(size)),
                ("Chunk overlap", _na(overlap)),
                ("BM25 indexed chunks", _bm25_indexed_count(status)),
                ("Embedding model", _na(status.get("embedding_model"))),
                ("Vectorstore path", _na(status.get("vectorstore_path"))),
                ("Knowledge base path", str(KNOWLEDGE_BASE_DIR)),
                ("Reranking", "Not enabled in this system"),
            ]
        )
    )

    tab_arch, tab_sim, tab_upload = st.tabs(
        ["Ingestion architecture", "Educational simulation", "Real ingestion"]
    )

    with tab_arch:
        render_html(render_simulation_tag("Architecture diagram"))
        _render_ingestion_architecture()
        st.write("")
        with st.expander("Text version of this diagram"):
            st.code(INGESTION_DIAGRAM, language="text")

    with tab_sim:
        _render_ingestion_simulation(pipeline)

    with tab_upload:
        _ingestion_tab_real(pipeline)


def _ingestion_tab_real(pipeline: RAGPipeline) -> None:
    st.markdown("**Real ingestion**")
    st.caption(
        "This uses the backend's existing incremental ingestion. Existing "
        "chunks are preserved and unchanged files are skipped. No "
        "ingestion logic is duplicated in the UI."
    )

    uploaded = st.file_uploader(
        "Upload a TXT or Markdown file",
        type=["txt", "md"],
        key="ingestion_page_uploader",
    )

    if uploaded is not None:
        if st.button("Run ingestion", key="ingestion_page_run", **WIDE):
            result = _ingest_uploaded_file(pipeline, uploaded, rerun=False)

            if result is not None:
                st.session_state.force_reload = True
                st.success("Ingestion completed.")

    result = st.session_state.get("last_ingestion")

    if not result:
        st.info("No ingestion has been run in this session yet.")
        return

    _render_ingestion_result(result)

    render_html(render_section_title("Stage replay"))
    render_html(render_simulation_tag("Pipeline visualization"))
    st.caption(
        "A replay of the stages drawn after ingestion finished. This is "
        "not live internal telemetry."
    )

    if st.button("Replay stages", key="ingestion_replay"):
        _replay_ingestion_stages(result)


# ==========================================================================
# RETRIEVAL LAB — SECTIONS
# ==========================================================================

def _lab_section_query(query: dict) -> None:
    render_html(render_section_title("Section A — Query"))

    render_html(
        render_trace_query_card(
            "ORIGINAL QUERY",
            query.get("original_query") or "Not available in current trace.",
        )
    )

    render_html(
        render_kv_rows(
            [
                ("Prepared query", _na(query.get("prepared_query"))),
                ("Normalization / aliases", _na(query.get("normalization"))),
                ("BM25 token count", _na(query.get("bm25_token_count"))),
                (
                    "Multi-intent detected",
                    "Yes" if query.get("is_multi_intent") else "No",
                ),
                (
                    "Greeting detected",
                    "Yes" if query.get("is_greeting") else "No",
                ),
            ]
        )
    )

    tokens = query.get("bm25_tokens") or []

    if tokens:
        with st.expander(f"BM25 tokens ({len(tokens)})"):
            st.code(" ".join(str(token) for token in tokens), language="text")
    else:
        st.caption("BM25 tokens: not available in current trace.")


def _lab_section_mode(retrieval: dict) -> None:
    render_html(render_section_title("Section B — Retrieval mode"))

    render_html(render_mode_chip(retrieval.get("mode_label", "Unknown")))

    render_html(
        render_kv_rows(
            [
                (
                    "Dense retrieval",
                    "Enabled"
                    if retrieval.get("dense_enabled")
                    else "Disabled",
                ),
                (
                    "BM25 retrieval",
                    "Enabled" if retrieval.get("bm25_enabled") else "Disabled",
                ),
                (
                    "RRF fusion",
                    "Enabled" if retrieval.get("rrf_enabled") else "Disabled",
                ),
                ("Candidate count", _na(retrieval.get("candidate_count"))),
                ("Fused count", _na(retrieval.get("fused_count"))),
                ("Dense results", _na(retrieval.get("dense_count"))),
                ("BM25 results", _na(retrieval.get("bm25_count"))),
                ("Vector top-k", _na(retrieval.get("vector_top_k"))),
                ("BM25 top-k", _na(retrieval.get("bm25_top_k"))),
                ("Hybrid top-k", _na(retrieval.get("hybrid_top_k"))),
                (
                    "Query embedding dimension",
                    _na(retrieval.get("query_embedding_dim")),
                ),
                (
                    "Reranking",
                    "Enabled"
                    if retrieval.get("reranker_enabled")
                    else "Disabled (by design)",
                ),
            ]
        )
    )


def _lab_section_dense(retrieval: dict) -> None:
    render_html(render_section_title("Vector retrieval — dense results"))

    rows = retrieval.get("raw_dense") or []

    if not rows:
        st.info("No dense results were recorded in this trace.")
        return

    _dataframe([
            {
                "Rank": row.get("rank"),
                "Source": row.get("filename"),
                "Chunk": row.get("chunk_id"),
                "Distance": row.get("distance"),
                "Score": row.get("score"),
            }
            for row in rows
        ])

    if any(row.get("distance") is not None for row in rows):
        st.caption(
            "Distance is the raw value returned by the vector store. "
            "Lower distance indicates greater vector similarity."
        )


def _lab_section_bm25(retrieval: dict) -> None:
    render_html(render_section_title("BM25 retrieval — lexical results"))

    rows = retrieval.get("raw_bm25") or []

    if not rows:
        st.info("No BM25 results were recorded in this trace.")
        return

    _dataframe([
            {
                "Rank": row.get("rank"),
                "Source": row.get("filename"),
                "Chunk": row.get("chunk_id"),
                "BM25 score": row.get("score"),
            }
            for row in rows
        ])

    st.caption(
        "BM25 scores are unbounded term-weighting scores, comparable "
        "within this result set only."
    )


def _lab_section_rrf(retrieval: dict) -> None:
    render_html(render_section_title("Reciprocal Rank Fusion"))

    k = retrieval.get("rrf_k", 60)

    render_html(
        render_formula_card(
            "RRF formula",
            "RRF(d) = Σ 1 / (k + rank)",
            f"Configured k = {k}",
        )
    )

    rows = retrieval.get("rrf_calculation") or []

    if not rows:
        st.info("No RRF calculations were recorded in this trace.")
        return

    _dataframe([
            {
                "Final rank": row.get("final_rank"),
                "Source": row.get("filename"),
                "Chunk": row.get("chunk_id"),
                "Dense rank": row.get("dense_rank"),
                "BM25 rank": row.get("bm25_rank"),
                "Dense contribution": row.get("dense_contribution"),
                "BM25 contribution": row.get("bm25_contribution"),
                "Final RRF": row.get("rrf_score"),
                "In both": "Yes" if row.get("agreement") else "No",
            }
            for row in rows
        ])

    st.caption(
        "A contribution of 0 means that retriever did not return the "
        "document at all."
    )

    with st.expander("Per-document arithmetic"):
        for row in rows:
            st.markdown(
                f"**{row.get('filename', 'unknown')} / "
                f"{row.get('chunk_id', 'unknown')}**"
            )

            lines = []

            if row.get("dense_formula"):
                lines.append(f"Dense: {row['dense_formula']}")
            else:
                lines.append("Dense: not retrieved (contribution 0)")

            if row.get("bm25_formula"):
                lines.append(f"BM25:  {row['bm25_formula']}")
            else:
                lines.append("BM25:  not retrieved (contribution 0)")

            if row.get("formula"):
                lines.append(f"RRF:   {row['formula']}")

            st.code("\n".join(lines), language="text")

    st.info(
        "RRF does not measure probability or confidence. It combines "
        "rankings from different retrieval systems so documents ranked "
        "highly by either system can be prioritized."
    )


def _lab_section_evidence(evidence: dict) -> None:
    render_html(render_section_title("Evidence evaluation"))

    render_html(
        render_evidence_badge(
            describe_evidence_level(evidence.get("level")),
            str(evidence.get("level") or "none"),
        )
    )

    render_html(
        render_kv_rows(
            [
                ("Evidence level", _na(evidence.get("level"))),
                (
                    "Should answer",
                    "Yes" if evidence.get("should_answer") else "No",
                ),
                ("Score source", _na(evidence.get("score_source"))),
                ("Supporting chunks", _na(evidence.get("supporting_chunks"))),
                (
                    "Retrieval agreement",
                    "Yes" if evidence.get("agreement") else "No",
                ),
                ("Top score", _na(evidence.get("top_score"))),
                ("Top gap", _na(evidence.get("top_gap"))),
                ("Top percentile", _na(evidence.get("top_percentile"))),
                ("Retrieval mode", _na(evidence.get("retrieval_mode"))),
                ("Reason", _na(evidence.get("reason"))),
            ]
        )
    )

    st.caption(
        "Confidence comes from this evidence layer, not from the RRF "
        "score. An RRF score is a rank-fusion weight and says nothing "
        "about how well the evidence supports an answer."
    )


def _lab_section_context(trace: dict, context: dict) -> None:
    render_html(render_section_title("Context sent to the model"))

    chunks = get_context_chunks(trace)

    col1, col2, col3 = st.columns(3)

    selected = context.get("selected_chunks")
    col1.metric("Chunks selected", selected if selected is not None else "N/A")

    chars = context.get("context_chars")
    col2.metric("Context characters", chars if chars is not None else "N/A")

    sources = context.get("unique_sources")

    if isinstance(sources, list):
        source_display = len(sources)
    elif isinstance(sources, int):
        source_display = sources
    else:
        source_display = "N/A"

    col3.metric("Sources represented", source_display)

    render_html(
        render_kv_rows(
            [
                ("Input chunks", _na(context.get("input_chunks"))),
                ("Removed chunks", _na(context.get("removed_chunks"))),
                ("Input characters", _na(context.get("input_chars"))),
                ("Selected characters", _na(context.get("selected_chars"))),
                ("Max chunks", _na(context.get("max_chunks"))),
                ("Max characters", _na(context.get("max_chars"))),
            ]
        )
    )

    if isinstance(sources, list) and sources:
        st.markdown(
            "**Sources represented:** "
            + ", ".join(str(item) for item in sources)
        )

    if not chunks:
        st.caption("Final context chunks are not available in this trace.")
        return

    with st.expander(f"Final context chunks ({len(chunks)})"):
        for chunk in chunks:
            st.markdown(
                f"**{chunk.get('position')}. "
                f"{chunk.get('filename') or 'unknown'}** · "
                f"`{chunk.get('chunk_id') or 'unknown'}` · "
                f"{chunk.get('chars')} characters"
            )
            st.code(chunk.get("text", ""), language="text")


def _lab_section_generation(generation: dict) -> None:
    render_html(render_section_title("Generation"))

    # The endpoint URL is deliberately not displayed: it identifies the
    # provider and is environment-specific.
    render_html(
        render_kv_rows(
            [
                ("Status", _na(generation.get("status"))),
                ("LLM model", _na(generation.get("model"))),
                (
                    "Endpoint",
                    "Configured" if generation.get("base_url") else None,
                ),
                ("Context characters", _na(generation.get("context_chars"))),
                ("Messages sent", _na(generation.get("message_count"))),
                ("Temperature", _na(generation.get("temperature"))),
                ("Response characters", _na(generation.get("response_chars"))),
                (
                    "Procedural answer",
                    "Yes" if generation.get("procedural") else "No",
                ),
            ]
        )
    )

    if generation.get("explanation"):
        st.caption(str(generation["explanation"]))

    if generation.get("response_preview"):
        with st.expander("Response preview"):
            st.code(str(generation["response_preview"]), language="text")

    if generation.get("error"):
        st.error(f"Generation error: {generation['error']}")




# ==========================================================================
# PAGE — CHUNK MONITOR
# ==========================================================================

def _render_chunk_monitor(pipeline: RAGPipeline) -> None:
    render_html(
        render_page_header(
            "DOCUMENT INSPECTION",
            "Chunk Monitor",
            "How each document is divided before it reaches the index.",
        )
    )

    files = list_txt_files(Path(KNOWLEDGE_BASE_DIR))

    if not files:
        st.info("No TXT or Markdown files were found in the knowledge base.")
        return

    chunks = _load_chunks(pipeline)

    if not chunks:
        st.warning(
            "Chunks could not be computed for preview. The index itself is "
            "unaffected."
        )
        return

    size, overlap = _chunk_config(pipeline)

    # ----------------------------------------------------------------
    # OVERVIEW
    # ----------------------------------------------------------------
    render_html(render_section_title("Knowledge base overview"))

    _dataframe([
            {
                "Document": path.name,
                "Chunks": len(
                    [c for c in chunks if c.get("filename") == path.name]
                ),
                "Characters": _document_characters(path),
            }
            for path in files
        ])

    # ----------------------------------------------------------------
    # DOCUMENT SELECTOR
    # ----------------------------------------------------------------
    render_html(render_section_title("Document"))

    selected_name = st.selectbox(
        "Select a document",
        [path.name for path in files],
    )

    selected_path = next(
        (path for path in files if path.name == selected_name),
        None,
    )

    selected_chunks = [
        c for c in chunks if c.get("filename") == selected_name
    ]

    if not selected_chunks:
        st.info("This document produced no non-empty chunks.")
        return

    file_size = None

    if selected_path is not None:
        try:
            file_size = selected_path.stat().st_size
        except Exception:  # noqa: BLE001
            file_size = None

    characters = (
        _document_characters(selected_path)
        if selected_path is not None
        else None
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "File size",
        f"{file_size:,} B" if file_size is not None else "N/A",
    )
    col2.metric(
        "Characters",
        f"{characters:,}" if characters is not None else "N/A",
    )
    col3.metric("Chunks", len(selected_chunks))
    col4.metric(
        "Chunk size / overlap",
        f"{size} / {overlap}" if size is not None else "N/A",
    )

    # ----------------------------------------------------------------
    # CHUNK BROWSER
    # ----------------------------------------------------------------
    render_html(render_section_title("Chunk browser"))

    chunk_number = st.number_input(
        "Chunk",
        min_value=1,
        max_value=len(selected_chunks),
        value=1,
        step=1,
    )

    index = int(chunk_number) - 1
    chunk = selected_chunks[index]

    render_html(render_chunk_map(len(selected_chunks), index))

    st.caption(
        f"Each cell is one chunk of {selected_name}, in document order. "
        "The highlighted cell is the chunk shown below."
    )

    text = chunk.get("text", "")

    render_html(
        render_kv_rows(
            [
                ("Chunk ID", _na(chunk.get("chunk_id"))),
                ("Filename", _na(chunk.get("filename"))),
                ("Characters", len(text)),
                ("Section heading", _na(chunk.get("section_heading"))),
                ("Chunk index", _na(chunk.get("chunk_index"))),
                ("Chunks in section", _na(chunk.get("total_section_chunks"))),
                ("Category", _na(chunk.get("category"))),
            ]
        )
    )

    st.code(text, language="text")

    with st.expander("Full chunk metadata"):
        st.json(
            {
                "source_path": chunk.get("source_path", ""),
                "section_heading": chunk.get("section_heading", ""),
                "section_path": chunk.get("section_path", []),
                "section_level": chunk.get("section_level", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_section_chunks": chunk.get("total_section_chunks", 0),
                "category": chunk.get("category", ""),
                "chunk_id": chunk.get("chunk_id", ""),
            }
        )

    # ----------------------------------------------------------------
    # OVERLAP EXPLANATION
    # ----------------------------------------------------------------
    render_html(render_section_title("Why chunks overlap"))

    if overlap:
        st.write(
            f"This knowledge base is configured with a {overlap}-character "
            f"overlap between neighbouring chunks of {size} characters. "
            "Overlap allows neighbouring chunks to retain contextual "
            "continuity, so a sentence that crosses a boundary is still "
            "readable in both chunks."
        )
    else:
        st.write(
            "Chunk overlap allows neighbouring chunks to retain contextual "
            "continuity. The configured overlap is not available from the "
            "pipeline, so no value is shown here."
        )

    with st.expander(f"All chunks in {selected_name}"):
        for position, item in enumerate(selected_chunks):
            st.markdown(
                f"**[ Chunk {position} ]** · "
                f"`{item.get('chunk_id', '')}` · "
                f"{len(item.get('text', ''))} characters"
            )
            st.code((item.get("text") or "")[:600], language="text")




# ============================================================================
# PAGE — SYSTEM ARCHITECTURE
# ============================================================================

def _render_architecture_page(
    pipeline: RAGPipeline,
    status: dict,
    collection_count,
) -> None:
    """
    Static architecture visualization of the real system.

    This page performs no retrieval and calls no model. The technical
    labels on the storage and model nodes come from live configuration,
    so the diagram cannot drift away from what is actually deployed.
    """
    render_html(
        render_page_header(
            "SYSTEM ARCHITECTURE",
            "Astronomy RAG System",
            "How knowledge enters the system, and how a question becomes "
            "a grounded answer.",
        )
    )

    render_html(render_simulation_tag("Architecture diagram — illustrative"))

    st.caption(
        "This diagram represents the components that actually exist in "
        "this project. It is not a live telemetry view."
    )

    render_html(render_arch_legend())

    st.write("")

    embedding_model = status.get("embedding_model") or "configured model"
    llm_model = _llm_model(status) or "configured model"

    chunks_label = (
        f"{collection_count} chunks"
        if collection_count is not None
        else "chunk count N/A"
    )

    size, overlap = _chunk_config(pipeline)

    chunk_tech = (
        f"size {size} / overlap {overlap}"
        if size is not None and overlap is not None
        else "size + overlap"
    )

    # ----------------------------------------------------------------
    # KNOWLEDGE PLANE
    # ----------------------------------------------------------------
    render_html(
        render_arch_plane_open(
            "Knowledge Plane — Ingestion",
            "Documents are loaded, cleaned and chunked once, then indexed "
            "twice: as vectors and as terms.",
        )
    )

    row = st.columns([1, 0.25, 1, 0.25, 1, 0.25, 1])

    with row[0]:
        render_html(
            render_arch_node(
                "📄",
                "Knowledge Sources",
                f"{status.get('txt_files_found', 'N/A')} TXT / MD files",
                "storage",
            )
        )

    with row[1]:
        render_html(render_arch_arrow("▶"))

    with row[2]:
        render_html(
            render_arch_node(
                "📥", "Document Loader", "read + detect", "processing"
            )
        )

    with row[3]:
        render_html(render_arch_arrow("▶"))

    with row[4]:
        render_html(
            render_arch_node(
                "🧹", "Text Processing", "normalize", "processing"
            )
        )

    with row[5]:
        render_html(render_arch_arrow("▶"))

    with row[6]:
        render_html(
            render_arch_node("✂", "Chunking", chunk_tech, "processing")
        )

    st.caption("Chunking fans out into two independent indexes:")

    split = st.columns(2)

    with split[0]:
        render_html(
            render_arch_node(
                "🧬", "Embedding Model", embedding_model, "processing"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "🗄", "Chroma Vector Store", chunks_label, "storage"
            )
        )

    with split[1]:
        render_html(
            render_arch_node(
                "🔎", "BM25", "term weighting", "retrieval"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "📚",
                "BM25 Index",
                "persistent lexical index",
                "storage",
            )
        )

    # ----------------------------------------------------------------
    # CONVERGENCE
    # ----------------------------------------------------------------
    bridge = st.columns(3)

    with bridge[1]:
        render_html(render_arch_arrow("▼"))
        st.caption("Both indexes serve the query plane below")
        render_html(render_arch_arrow("▼"))

    # ----------------------------------------------------------------
    # QUERY PLANE
    # ----------------------------------------------------------------
    render_html(
        render_arch_plane_open(
            "Query Plane — Retrieval & Generation",
            "A question is prepared once, searched by one or both "
            "retrievers depending on the selected mode, then gated, "
            "trimmed and answered.",
        )
    )

    head = st.columns(3)

    with head[1]:
        render_html(
            render_arch_node("👤", "User Query", "Mission Control", "processing")
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "⚙",
                "Query Preparation",
                "normalize / aliases",
                "processing",
            )
        )

    st.caption(
        "Vector mode uses dense search only. BM25 mode uses lexical "
        "search only. Hybrid mode runs both and fuses them."
    )

    search = st.columns(2)

    with search[0]:
        render_html(
            render_arch_node(
                "◈", "Dense Search", "Chroma nearest vectors", "retrieval"
            )
        )

    with search[1]:
        render_html(
            render_arch_node(
                "⌕", "BM25 Search", "lexical match", "retrieval"
            )
        )

    tail = st.columns(3)

    with tail[1]:
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "◉",
                "RRF Fusion",
                "Hybrid mode only",
                "retrieval",
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "⚖",
                "Evidence Gate",
                "relevance / support",
                "decision",
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "🗂",
                "Context Selection",
                "chunk + char budget",
                "processing",
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node("✦", "Generation", llm_model, "generation")
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "◆", "Answer", "response + sources", "generation"
            )
        )

    # ----------------------------------------------------------------
    # COMPONENT REFERENCE
    # ----------------------------------------------------------------
    render_html(render_section_title("Component reference"))

    left, right = st.columns(2)

    midpoint = (len(ARCH_DESCRIPTIONS) + 1) // 2

    for column, items in (
        (left, ARCH_DESCRIPTIONS[:midpoint]),
        (right, ARCH_DESCRIPTIONS[midpoint:]),
    ):
        with column:
            for name, description in items:
                with st.expander(name):
                    st.write(description)

    st.info(
        "Not present in this system: cross-encoder reranking is "
        "intentionally disabled, and there are no agents, external "
        "services or additional databases beyond those shown above."
    )

    with st.expander("Text version of the full flow"):
        st.code(INGESTION_DIAGRAM, language="text")
        st.code(RETRIEVAL_DIAGRAM, language="text")


# ============================================================================
# SIDEBAR
# ============================================================================

def _render_sidebar(
    pipeline: RAGPipeline,
    status: dict,
    collection_count,
    dev_embeddings: bool,
) -> str:
    """
    Sidebar: brand, grouped navigation, system status and index controls.

    Navigation lives here and only here. The native Streamlit collapse
    control is left untouched so the sidebar can always be reopened.
    """
    with st.sidebar:
        render_html(
            render_panel_card(
                "✦ ASTRONOMY OBSERVATORY",
                render_kv_rows([("Knowledge Core", "ONLINE")]),
            )
        )

        # ------------------------------------------------------------
        # NAVIGATION
        # ------------------------------------------------------------
        current = st.session_state.page

        if current not in PAGES:
            current = PAGE_MISSION

        for group_label, items in NAV_GROUPS:
            render_html(render_nav_group_label(group_label))

            for page_name, icon in items:
                active = page_name == current

                marker = "◉" if active else "◇"

                if st.button(
                    _sidebar_nav_label(marker, page_name, icon),
                    key=f"nav_{page_name}",
                    type="primary" if active else "secondary",
                    **WIDE,
                ):
                    if page_name != current:
                        st.session_state.page = page_name
                        st.rerun()

        render_html(render_sidebar_divider())

        # ------------------------------------------------------------
        # SYSTEM STATUS
        # ------------------------------------------------------------
        status_html = render_stat_grid(
            [
                (
                    str(collection_count)
                    if collection_count is not None
                    else "N/A",
                    "Indexed chunks",
                ),
                (str(status.get("txt_files_found", "N/A")), "Documents"),
            ]
        )

        status_html += render_status_row(
            "Vector Index", bool(status.get("index_available"))
        )
        status_html += render_status_row(
            "BM25 Index", bool(status.get("bm25_available"))
        )
        status_html += render_status_row(
            "Embeddings", bool(status.get("embedding_model"))
        )
        status_html += render_status_row(
            "LLM", bool(_llm_model(status))
        )

        render_html(render_panel_card("SYSTEM STATUS", status_html))

        meta_html = render_kv_rows(
            [
                ("Embedding Model", _na(status.get("embedding_model"))),
                ("LLM Model", _na(_llm_model(status))),
                ("Vector Store", _na(status.get("vectorstore_path"))),
            ]
        )

        if dev_embeddings:
            meta_html += render_kv_rows([("Mode", "Synthetic (DEV)")])

        render_html(render_panel_card("CONFIGURATION", meta_html))

        render_html(render_sidebar_divider())

        # ------------------------------------------------------------
        # INDEX CONTROL
        # ------------------------------------------------------------
        render_html(render_nav_group_label("Index Control"))

        if st.button("Refresh Status", key="ctl_refresh", **WIDE):
            st.session_state.force_reload = True
            st.rerun()

        if st.button("Sync Knowledge Base", key="ctl_sync", **WIDE):
            with st.spinner("Embedding any missing chunks …"):
                try:
                    result = pipeline.ingest_documents()

                    st.session_state.force_reload = True
                    st.session_state.ingestion_notice = {
                        "type": "success",
                        "message": (
                            f"Index updated: {result['chunks_indexed']} new "
                            f"chunks ({result.get('chunks_skipped', 0)} "
                            f"skipped, {result['collection_count']} total) "
                            f"from {result['files_processed']} files."
                        ),
                    }
                    st.rerun()

                except Exception as exc:  # noqa: BLE001
                    _service_error("INGESTION SERVICE UNAVAILABLE", exc)

        st.caption(
            "Incremental — existing embeddings are preserved, never "
            "rebuilt from scratch."
        )

        if st.button("Clear Chat", key="ctl_clear", **WIDE):
            st.session_state.pop("chat_history", None)
            st.session_state.last_response = None
            st.rerun()

    return st.session_state.page


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Astronomy Observatory — Mission Control",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_html(inject_global_css())

    dev_embeddings = os.getenv("DEV_EMBEDDINGS", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    Path(KNOWLEDGE_BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # SESSION DEFAULTS
    # ----------------------------------------------------------------
    st.session_state.setdefault("page", PAGE_MISSION)
    st.session_state.setdefault("retrieval_mode", "hybrid")
    st.session_state.setdefault("last_response", None)
    st.session_state.setdefault("last_ingestion", None)

    pipeline = _pipeline()
    status = _pipeline_status(pipeline)
    collection_count = _collection_count(pipeline)

    page = _render_sidebar(pipeline, status, collection_count, dev_embeddings)

    _show_ingestion_notice()

    if st.session_state.get("status_error"):
        _service_error(
            "SYSTEM STATUS UNAVAILABLE",
            st.session_state["status_error"],
        )

    if dev_embeddings:
        st.warning(
            "DEV MODE: synthetic embeddings are active. Semantic retrieval "
            "will not reflect real content similarity."
        )

    # Page dispatch. Only Mission Control can trigger retrieval or
    # generation; every other page is read-only.
    if page == PAGE_LAB:
        _render_retrieval_lab(st.session_state.last_response)
        return

    if page == PAGE_OBSERVATION:
        _render_observation_page(pipeline)
        return

    if page == PAGE_DASHBOARD:
        _render_dashboard_page(pipeline)
        return

    if page == PAGE_BENCHMARK:
        _render_benchmark_page(pipeline)
        return

    if page == PAGE_COMPARISON:
        _render_comparison_page(pipeline)
        return

    if page == PAGE_INGEST:
        _render_ingestion_page(pipeline, status, collection_count)
        return

    if page == PAGE_CHUNKS:
        _render_chunk_monitor(pipeline)
        return

    if page == PAGE_ARCH:
        _render_architecture_page(pipeline, status, collection_count)
        return

    _render_mission_control(pipeline, status, collection_count)


def _short_topic(query: str, index: int) -> str:
    words = [w for w in query.replace("?", "").split() if w.lower() not in {"what", "is", "are", "how", "does", "the", "and"}]
    return f"Q{index:02d} · {' '.join(words[:3]) or 'query'}"


def _render_observation_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Observation & Evaluation", "Persistent system insight dashboard. Live scores are heuristics, not accuracy."))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet.")
        return
    dense = sum(len(r.get("trace", {}).get("retrieval", {}).get("raw_dense", [])) for r in rows)
    bm25 = sum(len(r.get("trace", {}).get("retrieval", {}).get("raw_bm25", [])) for r in rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    cols = st.columns(4)
    cols[0].metric("Queries", len(rows)); cols[1].metric("Dense hits", dense); cols[2].metric("BM25 hits", bm25); cols[3].metric("Median latency", f"{latencies[len(latencies)//2]:.0f} ms")
    st.subheader("Retrieval and evidence trends")
    st.bar_chart({"Dense contribution": dense, "BM25 contribution": bm25})
    sources = {}
    for row in rows:
        for c in row.get("trace", {}).get("selected_evidence", []):
            sources[c.get("filename", "unknown")] = sources.get(c.get("filename", "unknown"), 0) + 1
    if sources:
        st.caption("Selected-evidence document distribution"); st.bar_chart(sources)
    latest = rows[0]; criteria = latest.get("evaluation", {}).get("criteria", {})
    agreement = criteria.get("retriever_agreement", {})
    direct = criteria.get("direct_answer_support", {})
    coverage = latest.get("answer_evaluation", {}).get("citation_coverage", {})
    st.subheader("Explainable AI insights")
    insights = [
        ("What happened", f"Dense and BM25 agreed on {agreement.get('metrics', {}).get('overlapping_chunks_count', 'no')} candidate(s).", agreement.get("reason", ""), "Agreement is consensus only; it does not establish correctness."),
        ("What happened", direct.get("reason", "Direct-answer support was not instrumented."), "Computed from the recorded candidate text, definitional patterns, term density, filename match, and term coverage.", "This distinguishes answer evidence from topical lexical overlap."),
        ("What happened", f"End-to-end latency was {latest.get('latency_ms')} ms.", "Only end-to-end request timing is recorded.", "Stage-wise latency is Not instrumented; no attribution is invented."),
        ("What happened", f"Citation coverage was {coverage.get('score', 'Not instrumented')}.", coverage.get("reason", "No citation result recorded."), "This is grounded-claim coverage, not answer accuracy."),
    ]
    for _, happened, why, meaning in insights:
        with st.expander(happened, expanded=True):
            st.write("**Why:**", why); st.write("**What it means:**", meaning)
    with st.expander("Query drill-down"):
        choices = {_short_topic(r["raw_query"], i): r for i, r in enumerate(rows, 1)}
        selected = choices[st.selectbox("Recorded query", list(choices))]
        st.json({"trace": selected.get("trace", {}), "evaluation": selected.get("evaluation", {}), "answer_evaluation": selected.get("answer_evaluation", {})})


def _render_dashboard_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("EVALUATION", "Evaluation Dashboard", "Calculation-visible live heuristic evaluation, separated from labelled benchmarks."))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet."); return
    data = {**rows[0].get("evaluation", {}).get("criteria", {}), **rows[0].get("answer_evaluation", {})}
    formulas = {
        "retrieval_strength": "60 + min(35, relative_gap×75), plus 5 when top candidate is in both retrievers; clamped 10–100.",
        "direct_answer_support": "Maximum top-6 candidate support: definitional 50 + up to 30, or density path 30 + up to 20; +15 filename match; +15×term coverage; clamped 5–100.",
        "evidence_relevance": "(direct×100 + broad×55 + tangential×10)/inspected, +10 when directly relevant evidence exists.",
        "retriever_agreement": "50% Jaccard(Dense,BM25) + 50% top-3 overlap ratio×100.",
        "ranking_stability": "Average top-3 Dense/BM25 survival in RRF top-5 ×100; clamped 20–100.",
        "evidence_sufficiency": "Simple: 95/75/40 based on direct support ≥70/≥45/else; complex: term coverage×70 + min(20,chunks×6).",
        "evidence_coherence": "Rule-based source concentration and source/query-term alignment.",
        "facet_coverage": "Evidence-covered detected live facets / detected facets ×100.",
        "evidence_claim_grounding": "Supported citations / citations ×100.",
        "citation_coverage": "Fully supported claims / extracted claims ×100.",
        "citation_correctness": "Citations with direct answer support / citations ×100.",
    }
    for name, metric in data.items():
        with st.expander(f"{metric.get('label', name.replace('_', ' ').title())}: {metric.get('score', 'Not instrumented')}"):
            st.write("**Definition:**", metric.get("definition", "Live answer/evidence check."))
            st.write("**Actual calculation:**", formulas.get(name, "Not instrumented"))
            st.write("**Data used:**", metric.get("metrics", metric.get("reason", "Recorded trace.")))
            st.write("**Interpretation:**", metric.get("reason", "Higher only reflects this project-specific heuristic."))
            st.caption("Standard vs heuristic: Project-specific heuristic, not accuracy.")
    st.subheader("Labelled benchmark evaluation")
    st.write("Precision@K = relevant retrieved / K; Recall@K = relevant retrieved / total relevant; MRR = 1 / rank of first relevant; nDCG@K = standard discounted gain. These require labels and are not live heuristic scores.")
    st.caption("Latency = end-to-end request duration. Median is available in Observation; P95 is Not instrumented until enough recorded observations exist.")


def _render_comparison_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Query Comparison", "Compare system behavior using compact query identifiers."))
    rows = load_observations(pipeline.vectorstore_path)
    if len(rows) < 2:
        st.info("Run at least two queries to compare observations."); return
    labels = [_short_topic(r["raw_query"], i) for i, r in enumerate(rows, 1)]
    first = rows[labels.index(st.selectbox("First query", labels, key="compare_a"))]
    second = rows[labels.index(st.selectbox("Second query", labels, index=1, key="compare_b"))]
    def metrics(row):
        c, a = row.get("evaluation", {}).get("criteria", {}), row.get("answer_evaluation", {})
        return {"Latency ms": row.get("latency_ms"), "Retrieval strength": c.get("retrieval_strength", {}).get("score"), "Agreement": c.get("retriever_agreement", {}).get("score"), "Evidence support": c.get("direct_answer_support", {}).get("score"), "Grounding": a.get("evidence_claim_grounding", {}).get("score"), "Citation coverage": a.get("citation_coverage", {}).get("score")}
    left, right = metrics(first), metrics(second)
    st.dataframe([{"Metric": key, labels[rows.index(first)]: left[key], labels[rows.index(second)]: right[key]} for key in left], hide_index=True, use_container_width=True)
    changed = [f"{key}: {left[key]} → {right[key]}" for key in left if left[key] != right[key]]
    st.subheader("What changed?"); st.write("; ".join(changed) if changed else "No recorded metric difference.")


def _render_benchmark_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("EVALUATION", "Benchmark", "Labelled benchmark evaluation only; intentionally separate from live heuristic evaluation."))
    st.write("A benchmark case supplies query, relevant_documents or relevant_chunks, expected_facets, and optional reference_answer.")
    st.write("When labels are present: Precision@K = relevant retrieved / K; Recall@K = relevant retrieved / total relevant; MRR = 1 / rank of first relevant; nDCG@K uses standard discounted cumulative gain normalized by ideal DCG.")
    st.info("No labelled benchmark result is stored yet. Live Retrieval Strength, Evidence Support, and Grounding are not labelled accuracy metrics.")


if __name__ == "__main__":
    main()
