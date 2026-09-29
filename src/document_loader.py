from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
    import docx.table
    import docx.text.paragraph
except ImportError:
    docx = None

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


SUPPORTED_EXTENSIONS: set[str] = {".txt", ".md", ".docx", ".pdf"}


def detect_file_type(
    path_or_name: str | Path,
    content: bytes | str | None = None,
) -> str:
    """Detect file format among 'txt', 'md', 'docx', 'pdf'.

    Checks extension first, with content magic-byte verification.
    """
    fname = str(path_or_name).lower()
    suffix = Path(fname).suffix.lower()

    if isinstance(content, bytes):
        if content.startswith(b"%PDF-"):
            return "pdf"
        if content.startswith(b"PK\x03\x04") and (b"word/" in content or suffix == ".docx"):
            return "docx"
    elif isinstance(content, str):
        if content.startswith("%PDF-"):
            return "pdf"

    if suffix in SUPPORTED_EXTENSIONS:
        return suffix.lstrip(".")

    path = Path(path_or_name)
    if path.exists() and path.is_file():
        try:
            with open(path, "rb") as f:
                header = f.read(2048)
            if header.startswith(b"%PDF-"):
                return "pdf"
            if header.startswith(b"PK\x03\x04") and b"word/" in header:
                return "docx"
        except Exception:
            pass

    return "txt"


def extract_text_from_txt(source: Path | str | bytes) -> str:
    """Extract clean text from a TXT file."""
    if isinstance(source, bytes):
        return source.decode("utf-8", errors="ignore")
    path = Path(source)
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_md(source: Path | str | bytes) -> str:
    """Extract clean text from a Markdown file."""
    if isinstance(source, bytes):
        return source.decode("utf-8", errors="ignore")
    path = Path(source)
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_docx(source: Path | str | bytes | io.BytesIO) -> str:
    """Extract clean text from a DOCX file preserving headings and tables."""
    if docx is None:
        raise ImportError("python-docx is required to extract DOCX files.")

    if isinstance(source, bytes):
        stream: Any = io.BytesIO(source)
    elif isinstance(source, (str, Path)):
        stream = str(source)
    elif isinstance(source, io.BytesIO):
        stream = source
    else:
        raise ValueError(f"Unsupported source type for DOCX: {type(source)}")

    doc = docx.Document(stream)
    text_parts: list[str] = []

    for child in doc.element.body:
        if child.tag.endswith("p"):
            p = docx.text.paragraph.Paragraph(child, doc)
            txt = p.text.strip()
            if not txt:
                continue
            style_name = getattr(getattr(p, "style", None), "name", "") or ""
            if style_name.startswith("Heading"):
                try:
                    level = int(style_name.split()[-1])
                except (ValueError, IndexError):
                    level = 2
                hashes = "#" * min(max(level, 1), 6)
                text_parts.append(f"\n{hashes} {txt}\n")
            else:
                text_parts.append(txt)
        elif child.tag.endswith("tbl"):
            t = docx.table.Table(child, doc)
            for row in t.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    text_parts.append(" | ".join(cells))

    if not text_parts and doc.paragraphs:
        for p in doc.paragraphs:
            txt = p.text.strip()
            if txt:
                text_parts.append(txt)

    return "\n\n".join(text_parts)


def extract_text_from_pdf(source: Path | str | bytes | io.BytesIO) -> list[tuple[int, str]]:
    """Extract page text from a PDF, returning [(1-based page_number, clean_text), ...]."""
    if pypdf is None:
        raise ImportError("pypdf is required to extract PDF files.")

    if isinstance(source, bytes):
        stream: Any = io.BytesIO(source)
    elif isinstance(source, (str, Path)):
        stream = open(source, "rb")
    elif isinstance(source, io.BytesIO):
        stream = source
    else:
        raise ValueError(f"Unsupported source type for PDF: {type(source)}")

    pages: list[tuple[int, str]] = []
    try:
        reader = pypdf.PdfReader(stream)
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            cleaned = clean_text(page_text)
            if cleaned:
                pages.append((idx + 1, cleaned))
    finally:
        if isinstance(source, (str, Path)):
            stream.close()

    return pages


def list_knowledge_files(base_dir: str | Path = KNOWLEDGE_BASE_DIR) -> list[Path]:
    """Return all supported knowledge files (.txt, .md, .docx, .pdf) under the directory."""
    folder = Path(base_dir)
    if not folder.exists():
        return []
    all_files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(p for p in folder.rglob(f"*{ext}") if p.is_file())
    return sorted(set(all_files))


def clean_text(raw_text: str) -> str:
    """Normalize whitespace while keeping meaningful formatting."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_document(
    path_or_name: Path | str,
    content: str | bytes | None = None,
) -> list[dict[str, Any]]:
    """Load and extract clean text from a file (TXT, MD, DOCX, or PDF).

    Returns a list of document dicts with keys:
    - filename: str
    - source_path: str
    - content: str
    - file_type: str ('txt' | 'md' | 'docx' | 'pdf')
    - page_number: int | None (1-based for PDF, None for non-PDF)
    """
    path = Path(path_or_name)
    filename = path.name
    source_path = str(path)

    raw_bytes: bytes | None = None
    if content is not None:
        if isinstance(content, bytes):
            raw_bytes = content
        else:
            raw_bytes = content.encode("utf-8")
    else:
        if not path.exists():
            return []
        try:
            raw_bytes = path.read_bytes()
        except OSError:
            return []

    file_type = detect_file_type(path_or_name, raw_bytes)
    documents: list[dict[str, Any]] = []

    if file_type == "pdf":
        source_input = raw_bytes if raw_bytes is not None else path
        pages = extract_text_from_pdf(source_input)
        for page_num, page_text in pages:
            documents.append(
                {
                    "filename": filename,
                    "source_path": source_path,
                    "content": page_text,
                    "file_type": "pdf",
                    "page_number": page_num,
                }
            )
    elif file_type == "docx":
        source_input = raw_bytes if raw_bytes is not None else path
        text = extract_text_from_docx(source_input)
        cleaned = clean_text(text)
        if cleaned:
            documents.append(
                {
                    "filename": filename,
                    "source_path": source_path,
                    "content": cleaned,
                    "file_type": "docx",
                    "page_number": None,
                }
            )
    elif file_type == "md":
        text = raw_bytes.decode("utf-8", errors="ignore") if raw_bytes else ""
        cleaned = clean_text(text)
        if cleaned:
            documents.append(
                {
                    "filename": filename,
                    "source_path": source_path,
                    "content": cleaned,
                    "file_type": "md",
                    "page_number": None,
                }
            )
    else:  # txt
        text = raw_bytes.decode("utf-8", errors="ignore") if raw_bytes else ""
        cleaned = clean_text(text)
        if cleaned:
            documents.append(
                {
                    "filename": filename,
                    "source_path": source_path,
                    "content": cleaned,
                    "file_type": "txt",
                    "page_number": None,
                }
            )

    return documents


def load_documents(base_dir: str | Path = KNOWLEDGE_BASE_DIR) -> list[dict[str, Any]]:
    """Load every knowledge file (.txt, .md, .docx, .pdf) and preserve source metadata."""
    documents: list[dict[str, Any]] = []
    for path in list_knowledge_files(base_dir):
        docs = load_document(path)
        documents.extend(docs)
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
        if re.match(r"^\d+(\.\d+)*\.\s+\S", ln) and not re.search(r"[.!?]\s*$", ln):
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
        file_type: str = "txt",
        page_number: int | None = None,
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
            if page_number is not None and page_number > 0:
                chunk_id = f"{filename}_p{page_number}_{section_index}_{idx}"
            else:
                chunk_id = f"{filename}_{section_index}_{idx}"

            out.append(
                {
                    "filename": filename,
                    "source_path": source_path,
                    "chunk_id": chunk_id,
                    "text": cleaned,
                    "section_heading": heading,
                    "section_path": section_path,
                    "section_level": len(section_path),
                    "category": category,
                    "has_heading": bool(heading),
                    "chunk_index": idx,
                    "total_section_chunks": 0,  # Filled below
                    "file_type": file_type,
                    "page_number": page_number if page_number is not None else 0,
                }
            )
        for chunk in out:
            chunk["total_section_chunks"] = len(out)
        return out

    for document in documents:
        text = document.get("content", "")
        filename = document["filename"]
        source_path = document.get("source_path", str(filename))
        file_type = document.get("file_type") or Path(filename).suffix.lstrip(".").lower() or "txt"
        page_number = document.get("page_number")
        if page_number is not None:
            try:
                page_number = int(page_number)
            except (ValueError, TypeError):
                page_number = None

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
            base_heading = re.sub(r"\.(txt|md|docx|pdf)$", "", filename, flags=re.I).replace("_", " ").title()
            if page_number is not None and page_number > 0:
                default_heading = f"{base_heading} Page {page_number}"
            else:
                default_heading = base_heading
            sections = [
                {
                    "heading": default_heading,
                    "text": text,
                    "path": [default_heading],
                }
            ]

        for sidx, sec in enumerate(sections):
            sec_chunks = _split_section(
                sec.get("text", ""),
                sec.get("heading", ""),
                filename,
                source_path,
                sidx,
                sec.get("path", []),
                file_type=file_type,
                page_number=page_number,
            )
            chunks.extend(sec_chunks)

    return chunks