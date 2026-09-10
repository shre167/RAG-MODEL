from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN = True
except Exception:
    RecursiveCharacterTextSplitter = None  # type: ignore
    _HAS_LANGCHAIN = False

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_BASE_DIR


# ============================================================
# ASTRONOMY DOMAIN CATEGORY KEYWORDS
# ============================================================

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Planetary Science": [
        "planet", "orbit", "atmosphere", "moon", "asteroid", "comet", "crater",
        "solar system", "mercury", "venus", "mars", "jupiter", "saturn",
        "uranus", "neptune", "dwarf", "pluto",
    ],
    "Stars & Astrophysics": [
        "star", "sun", "solar", "black hole", "singularity", "event horizon",
        "supernova", "neutron", "pulsar", "white dwarf", "red giant", "stellar",
        "fusion", "luminosity", "spectral", "main sequence",
    ],
    "Galaxies & Cosmology": [
        "galaxy", "galaxies", "milky way", "universe", "expansion", "big bang",
        "dark matter", "dark energy", "cosmic", "cosmology", "redshift",
        "quasar", "hubble constant",
    ],
    "Space Missions": [
        "isro", "chandrayaan", "pragyan", "vikram", "nasa", "apollo", "artemis",
        "voyager", "rover", "mission", "launch", "rocket", "lander", "orbiter",
        "probe", "iss", "spacex", "esa", "jaxa",
    ],
    "Space Technology": [
        "telescope", "jwst", "hubble", "satellite", "optics", "spectroscopy",
        "infrared", "launch vehicle", "thruster", "propulsion", "detector",
        "imaging", "radio telescope", "instrument",
    ],
}


def classify_document_by_filename(
    filename: str, content: str | None = None
) -> str | None:
    """Return category based on filename or content keywords."""
    fname = filename.lower()
    text = (content or "").lower()

    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in fname or kw in text)
        if score > 0:
            scores[category] = score

    return max(scores, key=scores.get) if scores else None  # type: ignore[arg-type]


def list_txt_files(base_dir: str | Path = KNOWLEDGE_BASE_DIR) -> list[Path]:
    """Return all .txt files under the knowledge_base directory."""
    folder = Path(base_dir)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*.txt") if p.is_file())


def list_knowledge_files(base_dir: str | Path = KNOWLEDGE_BASE_DIR) -> list[Path]:
    """Return all .txt and .md files under the knowledge_base directory."""
    folder = Path(base_dir)
    if not folder.exists():
        return []
    txt = [p for p in folder.rglob("*.txt") if p.is_file()]
    md = [p for p in folder.rglob("*.md") if p.is_file()]
    return sorted(txt + md)


def clean_text(raw_text: str) -> str:
    """Normalize whitespace while keeping meaningful formatting."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_documents(base_dir: str | Path = KNOWLEDGE_BASE_DIR) -> list[dict[str, Any]]:
    """Load every textual knowledge file and preserve source metadata."""
    documents: list[dict[str, Any]] = []
    for path in list_knowledge_files(base_dir):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        cleaned = clean_text(content)
        if not cleaned:
            continue
        documents.append(
            {
                "filename": path.name,
                "source_path": str(path),
                "content": cleaned,
            }
        )
    return documents


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split documents into overlapping chunks with section headings prepended.

    Heading detection supports:
    - Numbered section patterns:  "1. OVERVIEW", "2.3. MOONS"
    - Markdown-style headings:    "# Title", "## Subsection"
    - ALL-CAPS lines ≥ 2 words, ≤ 120 chars
    - Lines ending with ':' followed by a blank line
    """
    chunks: list[dict[str, Any]] = []

    def _is_heading(
        line: str,
        next_line: str | None = None,
        prev_line: str | None = None,
    ) -> bool:
        ln = line.strip()
        if not ln or len(ln) < 3:
            return False

        # Markdown headings (#, ##, ###)
        if re.match(r"^#{1,4}\s+\S", ln):
            return True

        # Numbered section headings: "1.", "2.3.", "10."
        if re.match(r"^\d+(\.\d+)*\.\s+\S", ln):
            return True

        # ALL CAPS substantial headings (≥ 2 words, ≤ 120 chars)
        if ln.isupper() and 8 < len(ln) <= 120 and len(ln.split()) >= 2:
            return True

        # Colon-ended lines followed by a blank line
        if ln.endswith(":") and next_line is not None and not next_line.strip():
            return True

        # Common section markers in astronomy / science text files
        markers = [
            "OVERVIEW", "INTRODUCTION", "HISTORY", "DISCOVERY",
            "EXPLORATION", "PHYSICAL PROPERTIES", "COMPOSITION",
            "ATMOSPHERE", "SURFACE", "STRUCTURE", "FORMATION",
            "CHARACTERISTICS", "MISSIONS", "OBSERVATIONS",
            "CLASSIFICATION", "TYPES", "REFERENCES", "SUMMARY",
            "SECTION", "CHAPTER", "APPENDIX",
        ]
        if any(m in ln.upper() for m in markers) and len(ln.split()) <= 8:
            return True

        return False

    def _strip_heading_marker(line: str) -> str:
        ln = line.strip()
        # Remove markdown #
        ln = re.sub(r"^#{1,4}\s+", "", ln)
        # Remove trailing colon
        ln = ln.rstrip(":")
        return ln.strip()

    def _split_section(
        text: str,
        heading: str,
        filename: str,
        source_path: str,
        section_index: int,
        section_path: list[str],
    ) -> list[dict[str, Any]]:
        heading_prefix = f"## {heading}\n\n" if heading else ""
        text_with_heading = heading_prefix + text

        if _HAS_LANGCHAIN and RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            segments = splitter.split_text(text_with_heading)
        else:
            def _simple_split(t: str, size: int, overlap: int) -> list[str]:
                if len(t) <= size:
                    return [t]
                parts: list[str] = []
                start = 0
                step = max(1, size - overlap)
                while start < len(t):
                    end = start + size
                    parts.append(t[start:end])
                    if end >= len(t):
                        break
                    start += step
                return parts

            segments = _simple_split(text_with_heading, chunk_size, chunk_overlap)

        out: list[dict[str, Any]] = []
        for idx, seg in enumerate(segments):
            cleaned = clean_text(seg)
            if not cleaned or len(cleaned) < 10:
                continue
            category = classify_document_by_filename(filename, cleaned)
            out.append(
                {
                    "filename": filename,
                    "source_path": source_path,
                    "chunk_id": f"{filename}_{section_index}_{idx}",
                    "text": cleaned,
                    "section_heading": heading,
                    "section_path": section_path,
                    "section_level": len(section_path),
                    "category": category,
                    "has_heading": bool(heading),
                    "chunk_index": idx,
                    "total_section_chunks": 0,  # Filled below
                }
            )
        for chunk in out:
            chunk["total_section_chunks"] = len(out)
        return out

    for document in documents:
        text = document.get("content", "")
        filename = document["filename"]
        lines = text.splitlines()

        sections: list[dict[str, Any]] = []
        current_heading = ""
        current_buf: list[str] = []
        section_path: list[str] = []

        for i, line in enumerate(lines):
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            prev_line = lines[i - 1] if i > 0 else None

            if _is_heading(line, next_line, prev_line):
                if current_buf:
                    sections.append(
                        {
                            "heading": current_heading,
                            "text": "\n".join(current_buf),
                            "path": section_path.copy(),
                        }
                    )
                    current_buf = []
                current_heading = _strip_heading_marker(line)
                section_path = [current_heading]
            else:
                current_buf.append(line)

        if current_buf:
            sections.append(
                {
                    "heading": current_heading,
                    "text": "\n".join(current_buf),
                    "path": section_path.copy(),
                }
            )

        if not sections:
            sections = [
                {
                    "heading": filename.replace(".txt", "").replace(".md", "").replace("_", " ").title(),
                    "text": text,
                    "path": [filename],
                }
            ]

        for sidx, sec in enumerate(sections):
            sec_chunks = _split_section(
                sec.get("text", ""),
                sec.get("heading", ""),
                filename,
                document["source_path"],
                sidx,
                sec.get("path", []),
            )
            chunks.extend(sec_chunks)

    return chunks