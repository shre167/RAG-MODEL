from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

from src.document_loader import list_knowledge_files
from src.rag_pipeline.models import KBState

logger = logging.getLogger(__name__)


def _kb_state_file(vectorstore_path: Path) -> Path:
    return vectorstore_path / "kb_state.json"


def load_kb_state(vectorstore_path: Path) -> dict[str, Any]:
    state_file = _kb_state_file(vectorstore_path)
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read kb_state.json: %s", exc)
    return {
        "version": 1,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def save_kb_state(vectorstore_path: Path, data: dict[str, Any]) -> None:
    state_file = _kb_state_file(vectorstore_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save kb_state.json: %s", exc)


def get_current_kb_state(pipeline: Any) -> KBState:
    """
    Query the live state of the knowledge base, Chroma index, and BM25 index.
    Checks consistency between Dense and Lexical representations.
    """
    kb_path = Path(pipeline.knowledge_base_path)
    vs_path = Path(pipeline.vectorstore_path)

    # Count source documents
    doc_count = 0
    if kb_path.exists():
        try:
            doc_count = len(list_knowledge_files(kb_path))
        except Exception:
            doc_count = 0

    # Count Chroma collection vectors
    chroma_count = 0
    try:
        chroma_count = pipeline.vector_store.get_collection_count()
    except Exception as exc:
        logger.warning("Could not read collection count: %s", exc)
        chroma_count = 0

    # Count BM25 indexed chunks
    bm25_count = 0
    if pipeline.bm25 is not None:
        try:
            bm25_count = len(getattr(pipeline.bm25, "_chunks", []))
        except Exception:
            bm25_count = 0

    # Consistency evaluation
    if chroma_count == bm25_count:
        indexes_consistent = True
        consistency_message = (
            f"✓ Indexes consistent: {chroma_count} chunks in Chroma "
            f"and {bm25_count} chunks in BM25"
        )
    else:
        indexes_consistent = False
        consistency_message = (
            f"⚠ INDEX MISMATCH: Chroma has {chroma_count} chunks, "
            f"BM25 has {bm25_count} chunks. Indexes may be out of sync."
        )

    stored_data = load_kb_state(vs_path)
    version = stored_data.get("version", 1)
    last_updated = stored_data.get(
        "last_updated", datetime.now(timezone.utc).isoformat()
    )

    return KBState(
        version=version,
        document_count=doc_count,
        chroma_chunks=chroma_count,
        bm25_chunks=bm25_count,
        indexes_consistent=indexes_consistent,
        last_updated=last_updated,
        consistency_message=consistency_message,
    )


def increment_kb_version(pipeline: Any) -> KBState:
    """
    Increment the KB version counter after document addition, update, or deletion.
    """
    vs_path = Path(pipeline.vectorstore_path)
    data = load_kb_state(vs_path)
    new_version = int(data.get("version", 0)) + 1
    now_iso = datetime.now(timezone.utc).isoformat()
    data["version"] = new_version
    data["last_updated"] = now_iso
    save_kb_state(vs_path, data)
    return get_current_kb_state(pipeline)


def check_index_consistency(pipeline: Any) -> dict[str, Any]:
    """Check whether Chroma and BM25 indexes represent the same chunk count."""
    state = get_current_kb_state(pipeline)
    return {
        "consistent": state.indexes_consistent,
        "chroma_chunks": state.chroma_chunks,
        "bm25_chunks": state.bm25_chunks,
        "message": state.consistency_message,
        "version": state.version,
    }


def delete_document(
    pipeline: Any,
    filename: str,
    delete_file_from_disk: bool = True,
) -> dict[str, Any]:
    """
    Safely delete all chunks belonging to a document from both Chroma and BM25.
    Prevents stale chunks from lingering in retrieval indexes.
    """
    filename = Path(filename).name
    vs_path = Path(pipeline.vectorstore_path)
    manifest_path = vs_path / "ingestion_manifest.json"

    # 1. Load manifest to find recorded chunk IDs
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    file_info = manifest.get(filename, {})
    chunk_ids = list(file_info.get("chunk_ids", []))

    chroma_removed = 0
    # 2. Delete from Chroma
    if chunk_ids:
        try:
            pipeline.vector_store.collection.delete(ids=chunk_ids)
            chroma_removed = len(chunk_ids)
            logger.info("Chroma: deleted %d chunks by ID for %s", len(chunk_ids), filename)
        except Exception as exc:
            logger.warning("Chroma chunk deletion by ID failed for %s: %s", filename, exc)

    # Fallback delete by metadata where supported
    try:
        pipeline.vector_store.collection.delete(where={"filename": filename})
    except Exception:
        pass

    # 3. Delete from BM25
    bm25_removed = 0
    if pipeline.bm25 is not None:
        try:
            bm25_removed = pipeline.bm25.delete_file(filename)
            pipeline.bm25.save_index()
            logger.info("BM25: deleted %d chunks for %s", bm25_removed, filename)
        except Exception as exc:
            logger.warning("BM25 deletion failed for %s: %s", filename, exc)

    # 4. Remove from manifest
    if filename in manifest:
        manifest.pop(filename, None)
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not update manifest after deletion: %s", exc)

    # 5. Optionally remove file from disk
    file_deleted = False
    if delete_file_from_disk:
        disk_path = Path(pipeline.knowledge_base_path) / filename
        if disk_path.exists():
            try:
                disk_path.unlink()
                file_deleted = True
                logger.info("Deleted %s from knowledge base disk directory.", filename)
            except Exception as exc:
                logger.warning("Could not delete file %s from disk: %s", disk_path, exc)

    # 6. Update KB version and state
    new_state = increment_kb_version(pipeline)

    return {
        "filename": filename,
        "deleted_from_chroma": chroma_removed,
        "deleted_from_bm25": bm25_removed,
        "deleted_from_disk": file_deleted,
        "collection_count": new_state.chroma_chunks,
        "bm25_chunks": new_state.bm25_chunks,
        "kb_version": new_state.version,
        "indexes_consistent": new_state.indexes_consistent,
    }
