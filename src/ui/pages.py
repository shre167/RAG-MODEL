from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import VECTORSTORE_DIR
from src.rag_pipeline import RAGPipeline
from src.rag_pipeline.observation_store import load_observations
from src.ui.astronomy_theme import render_html, render_page_header, render_section_title


def render_observation_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Observation & Evaluation", "Dashboard"))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet.")
        return
    st.metric("Total Queries", len(rows))


def render_dashboard_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("EVALUATION", "Evaluation Dashboard", "Live metrics"))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet.")
        return
    st.metric("Total Queries", len(rows))


def render_comparison_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Query Comparison", "Compare queries"))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet.")


def _normalize_chunk(c: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Ensure all required canonical metadata fields are present."""
    text = str(c.get("text") or "").strip()
    filename = str(c.get("filename") or c.get("book_name") or c.get("source_name") or "untitled_document.pdf")
    book_title = str(c.get("book_title") or c.get("book") or c.get("source_label") or "Untitled document")
    if book_title.lower().endswith(".pdf"):
        book_title = book_title[:-4].replace(" (1)", "").strip()
    if not book_title.strip():
        book_title = "Untitled document"

    ch_num = c.get("chapter_num")
    try:
        ch_int = int(ch_num) if ch_num is not None and int(ch_num) > 0 else 0
    except (ValueError, TypeError):
        ch_int = 0

    p_start = c.get("page_start") or c.get("page_number") or 0
    p_end = c.get("page_end") or p_start
    try:
        p_start_int = int(p_start)
    except (ValueError, TypeError):
        p_start_int = 0
    try:
        p_end_int = int(p_end)
    except (ValueError, TypeError):
        p_end_int = p_start_int

    sec = str(c.get("section_heading") or c.get("section_title") or c.get("section") or "").strip()

    return {
        "chunk_id": str(c.get("chunk_id") or f"chunk_{index}"),
        "chunk_index": int(c.get("chunk_index") if c.get("chunk_index") is not None else index),
        "book": book_title,
        "book_title": book_title,
        "book_name": filename,
        "filename": filename,
        "book_id": str(c.get("book_id") or "untitled_document_1"),
        "chapter_num": ch_int,
        "chapter": ch_int if ch_int > 0 else None,
        "chapter_title": str(c.get("chapter_title") or "").strip(),
        "section_heading": sec,
        "section_title": sec,
        "section": sec or "General",
        "page_start": p_start_int,
        "page_end": p_end_int,
        "page_range": f"{p_start_int}–{p_end_int}" if p_end_int > p_start_int else str(p_start_int),
        "token_count": int(c.get("token_count") or len(text.split())),
        "character_count": len(text),
        "prev_chunk_id": str(c.get("prev_chunk_id") or ""),
        "next_chunk_id": str(c.get("next_chunk_id") or ""),
        "text": text,
        "raw": c,
    }


def render_chunk_explorer(chunks: list[dict[str, Any]], pipeline: Any = None) -> None:
    """
    Redesigned Chunk Explorer organized around book hierarchy.
    Exposes canonical summary, chapter distribution bar chart,
    hierarchical Chapter -> Section -> Chunks accordion, and
    detailed individual chunk inspection with boundary context.
    """
    # Fallback to cache or Chroma if chunks empty
    if not chunks:
        cache_path = Path(VECTORSTORE_DIR) / "canonical_chunks_cache.json"
        if cache_path.exists():
            try:
                chunks = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                chunks = []

    if not chunks and pipeline and hasattr(pipeline, "vector_store"):
        try:
            data = pipeline.vector_store.collection.get(include=["documents", "metadatas"])
            if data and data.get("ids"):
                chunks = []
                for i, cid in enumerate(data["ids"]):
                    m = data["metadatas"][i] if i < len(data["metadatas"]) else {}
                    doc = data["documents"][i] if i < len(data["documents"]) else ""
                    chunks.append({**m, "chunk_id": cid, "text": doc})
        except Exception:
            chunks = []

    if not chunks:
        st.warning("No canonical chunks are currently available. Ingest documents to inspect.")
        return

    # Normalize all chunks
    normalized_chunks = [_normalize_chunk(c, i) for i, c in enumerate(chunks)]
    normalized_chunks.sort(key=lambda x: x["chunk_index"])
    chunk_by_id = {c["chunk_id"]: c for c in normalized_chunks}

    # =========================================================================
    # 1. SUMMARY
    # =========================================================================
    render_html(render_section_title("Canonical Chunks Summary"))

    total_chunks = len(normalized_chunks)
    chapters = sorted({c["chapter_num"] for c in normalized_chunks if c["chapter_num"] > 0})
    sections = {c["section"] for c in normalized_chunks if c["section"] and c["section"] != "General"}
    tokens = [c["token_count"] for c in normalized_chunks]
    avg_tokens = round(sum(tokens) / len(tokens)) if tokens else 0
    min_tokens = min(tokens) if tokens else 0
    max_tokens = max(tokens) if tokens else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Chunks", total_chunks)
    c2.metric("Chapters", len(chapters))
    c3.metric("Sections", len(sections))
    c4.metric("Average Tokens", avg_tokens)
    c5.metric("Minimum Tokens", min_tokens)
    c6.metric("Maximum Tokens", max_tokens)

    # =========================================================================
    # TOKEN DISTRIBUTION
    # =========================================================================
    render_html(render_section_title("Token Distribution"))

    token_df = pd.DataFrame({"tokens": tokens})

    col_dist, col_stats = st.columns([7, 3])
    with col_dist:
        st.bar_chart(token_df["tokens"].value_counts().sort_index(), use_container_width=True)

    with col_stats:
        st.markdown("**Token Statistics**")
        median_tokens = sorted(tokens)[len(tokens) // 2] if tokens else 0
        st.metric("Average", f"{avg_tokens} tokens")
        st.metric("Median",  f"{median_tokens} tokens")
        st.metric("Minimum", f"{min_tokens} tokens")
        st.metric("Maximum", f"{max_tokens} tokens")
        # Identify outliers
        small = sum(1 for t in tokens if t < 80)
        large = sum(1 for t in tokens if t > 450)
        if small:
            st.warning(f"{small} chunks below 80 tokens")
        if large:
            st.warning(f"{large} chunks above 450 tokens")

    # =========================================================================
    # CHUNK CONSISTENCY DASHBOARD
    # =========================================================================
    render_html(render_section_title("Chunk Consistency"))

    total = len(normalized_chunks)
    empty = sum(1 for c in normalized_chunks if not c["text"].strip())
    dup_ids = total - len({c["chunk_id"] for c in normalized_chunks})
    dup_text = total - len({c["text"][:200] for c in normalized_chunks if c["text"]})
    missing_chapter = sum(1 for c in normalized_chunks if not c["chapter_num"])
    missing_section = sum(1 for c in normalized_chunks if not c["section_heading"])
    missing_page = sum(1 for c in normalized_chunks if not c["page_start"])
    below_min = sum(1 for c in normalized_chunks if c["token_count"] < 80)
    above_max = sum(1 for c in normalized_chunks if c["token_count"] > 450)

    # Broken prev/next links
    all_ids = {c["chunk_id"] for c in normalized_chunks}
    broken_prev = sum(1 for c in normalized_chunks if c["prev_chunk_id"] and c["prev_chunk_id"] not in all_ids)
    broken_next = sum(1 for c in normalized_chunks if c["next_chunk_id"] and c["next_chunk_id"] not in all_ids)

    # Chapter-boundary violations: chunk with chapter_num different from prev chunk
    boundary_violations = 0
    for i in range(1, len(normalized_chunks)):
        prev_ch = normalized_chunks[i-1]["chapter_num"]
        curr_ch = normalized_chunks[i]["chapter_num"]
        prev_next = normalized_chunks[i-1]["next_chunk_id"]
        curr_prev = normalized_chunks[i]["prev_chunk_id"]
        # A violation is a chapter change where the link still points across
        if prev_ch != curr_ch and prev_next and curr_prev:
            boundary_violations += 1

    issues = [
        ("Empty chunks", empty, empty > 0),
        ("Duplicate chunk IDs", dup_ids, dup_ids > 0),
        ("Duplicate chunk text", dup_text, dup_text > 0),
        ("Missing chapter", missing_chapter, missing_chapter > 0),
        ("Missing section", missing_section, missing_section > 0),
        ("Missing page", missing_page, missing_page > 0),
        ("Below min tokens (80)", below_min, below_min > 0),
        ("Above max tokens (450)", above_max, above_max > 0),
        ("Broken prev links", broken_prev, broken_prev > 0),
        ("Broken next links", broken_next, broken_next > 0),
        ("Chapter-boundary links", boundary_violations, boundary_violations > 0),
    ]

    has_issues = any(flag for _, _, flag in issues)
    if not has_issues:
        st.success(f"✅ All {total} chunks pass consistency checks.")
    else:
        warn_cols = st.columns(4)
        for idx, (label, count, is_issue) in enumerate(issues):
            col = warn_cols[idx % 4]
            if is_issue:
                col.metric(label, count, delta=f"⚠️ issue", delta_color="inverse")
            else:
                col.metric(label, count, delta="✅ OK", delta_color="off")

    # =========================================================================
    # 2. CHAPTER DISTRIBUTION
    # =========================================================================
    render_html(render_section_title("Chapter Chunk Distribution"))

    chapter_dist = {}
    for ch in chapters:
        chapter_dist[f"Chapter {ch}"] = sum(1 for c in normalized_chunks if c["chapter_num"] == ch)

    col_chart, col_stats = st.columns([7, 3])
    with col_chart:
        chart_df = pd.DataFrame(
            [{"Chapter": k, "Chunks": v} for k, v in chapter_dist.items()]
        ).set_index("Chapter")
        st.bar_chart(chart_df, use_container_width=True)

    with col_stats:
        st.markdown("**Chunks Per Chapter**")
        for k, v in chapter_dist.items():
            pct = round((v / total_chunks) * 100, 1) if total_chunks else 0
            st.markdown(f"• **{k}**: {v} chunks `({pct}%)`")

    # =========================================================================
    # 3. HIERARCHICAL STRUCTURE (Chapter -> Section -> Chunks)
    # =========================================================================
    render_html(render_section_title("Book Hierarchy (Chapter → Section → Chunks)"))
    st.caption("Expand any chapter or section below to explore its constituent chunks.")

    # Manage selected chunk in session state
    if "selected_chunk_id" not in st.session_state:
        st.session_state.selected_chunk_id = normalized_chunks[0]["chunk_id"]

    for ch in chapters:
        ch_chunks = [c for c in normalized_chunks if c["chapter_num"] == ch]
        ch_title = next((c["chapter_title"] for c in ch_chunks if c["chapter_title"]), "")
        title_str = f" — {ch_title}" if ch_title else ""

        with st.expander(f"📖 Chapter {ch}{title_str}  ({len(ch_chunks)} chunks)", expanded=(ch == 1)):
            # Group by section
            section_dict: dict[str, list[dict[str, Any]]] = {}
            for c in ch_chunks:
                sec_name = c["section"] or "General / Unnamed Section"
                section_dict.setdefault(sec_name, []).append(c)

            for sec_name, sec_chunks in section_dict.items():
                st.markdown(f"##### 📑 Section: {sec_name} `({len(sec_chunks)} chunks)`")

                chunk_cols = st.columns(min(4, max(1, len(sec_chunks))))
                for idx, sc in enumerate(sec_chunks):
                    col = chunk_cols[idx % len(chunk_cols)]
                    is_active = (sc["chunk_id"] == st.session_state.selected_chunk_id)
                    btn_label = f"Chunk #{sc['chunk_index']:03d} (pp. {sc['page_range']})"
                    btn_type = "primary" if is_active else "secondary"
                    if col.button(btn_label, key=f"btn_chunk_{sc['chunk_id']}", type=btn_type, use_container_width=True):
                        st.session_state.selected_chunk_id = sc["chunk_id"]
                        st.rerun()

    # =========================================================================
    # 4. CHUNK INSPECTION & DETAILS PANEL
    # =========================================================================
    render_html(render_section_title("Chunk Details & Context Inspector"))

    # Direct chunk selector dropdown for convenience
    chunk_options = {
        f"Chunk {c['chunk_index']:03d} | Ch. {c['chapter_num']} | pp. {c['page_range']} | {c['section'][:30]}": c["chunk_id"]
        for c in normalized_chunks
    }
    opt_labels = list(chunk_options.keys())
    opt_ids = list(chunk_options.values())
    cur_idx = opt_ids.index(st.session_state.selected_chunk_id) if st.session_state.selected_chunk_id in opt_ids else 0

    selected_label = st.selectbox(
        "Direct Chunk Selector",
        options=opt_labels,
        index=cur_idx,
        key="direct_chunk_selector",
    )
    st.session_state.selected_chunk_id = chunk_options[selected_label]
    selected_chunk = chunk_by_id.get(st.session_state.selected_chunk_id, normalized_chunks[0])

    # Display clear Chunk Details Panel
    det_c1, det_c2, det_c3, det_c4 = st.columns(4)
    det_c1.metric("Book", selected_chunk["book_title"])
    det_c2.metric("Chapter", f"Chapter {selected_chunk['chapter_num']}" if selected_chunk['chapter_num'] else "None")
    det_c3.metric("Section", selected_chunk["section"][:25] + ("..." if len(selected_chunk["section"]) > 25 else ""))
    det_c4.metric("Page Range", f"Pages {selected_chunk['page_range']}")

    det_c5, det_c6, det_c7, det_c8 = st.columns(4)
    det_c5.metric("Chunk Index", f"#{selected_chunk['chunk_index']}")
    det_c6.metric("Token Count", f"{selected_chunk['token_count']} tokens")
    det_c7.metric("Character Count", f"{selected_chunk['character_count']} chars")
    det_c8.metric("PDF Filename", selected_chunk["filename"][:20] + ("..." if len(selected_chunk["filename"]) > 20 else ""))

    st.markdown(f"**Chunk ID:** `{selected_chunk['chunk_id']}`")
    if selected_chunk["chapter_title"]:
        st.markdown(f"**Chapter Title:** {selected_chunk['chapter_title']}")
    if selected_chunk["section_heading"]:
        st.markdown(f"**Section Heading:** {selected_chunk['section_heading']}")

    # Full Chunk Text
    st.markdown("**Full Chunk Text:**")
    st.code(selected_chunk["text"], language="text")

    # =========================================================================
    # 5. BOUNDARY VIEW (Previous chunk → Current chunk → Next chunk)
    # =========================================================================
    render_html(render_section_title("Boundary Context View (Previous → Current → Next)"))

    b_prev, b_curr, b_next = st.columns(3)

    prev_id = selected_chunk.get("prev_chunk_id")
    prev_chunk = chunk_by_id.get(prev_id) if prev_id else None

    next_id = selected_chunk.get("next_chunk_id")
    next_chunk = chunk_by_id.get(next_id) if next_id else None

    with b_prev:
        st.markdown("### ⬅️ Previous Chunk")
        if prev_chunk:
            st.markdown(f"**Chunk #{prev_chunk['chunk_index']:03d}** (pp. {prev_chunk['page_range']})")
            st.caption(f"{prev_chunk['token_count']} tokens | Ch. {prev_chunk['chapter_num']}")
            st.code((prev_chunk["text"][:280] + "...") if len(prev_chunk["text"]) > 280 else prev_chunk["text"], language="text")
            if st.button("⬅️ Navigate to Previous", key="nav_to_prev", use_container_width=True):
                st.session_state.selected_chunk_id = prev_chunk["chunk_id"]
                st.rerun()
        else:
            st.info("None (Beginning of Document)")

    with b_curr:
        st.markdown("### 🎯 Current Chunk")
        st.markdown(f"**Chunk #{selected_chunk['chunk_index']:03d}** (pp. {selected_chunk['page_range']})")
        st.caption(f"{selected_chunk['token_count']} tokens | Ch. {selected_chunk['chapter_num']}")
        st.code((selected_chunk["text"][:280] + "...") if len(selected_chunk["text"]) > 280 else selected_chunk["text"], language="text")
        st.success("Currently Inspected")

    with b_next:
        st.markdown("### ➡️ Next Chunk")
        if next_chunk:
            st.markdown(f"**Chunk #{next_chunk['chunk_index']:03d}** (pp. {next_chunk['page_range']})")
            st.caption(f"{next_chunk['token_count']} tokens | Ch. {next_chunk['chapter_num']}")
            st.code((next_chunk["text"][:280] + "...") if len(next_chunk["text"]) > 280 else next_chunk["text"], language="text")
            if st.button("➡️ Navigate to Next", key="nav_to_next", use_container_width=True):
                st.session_state.selected_chunk_id = next_chunk["chunk_id"]
                st.rerun()
        else:
            st.info("None (End of Document)")

    # Optional Expandable Raw JSON View
    with st.expander("Raw Metadata / JSON View", expanded=False):
        st.json(selected_chunk["raw"])
