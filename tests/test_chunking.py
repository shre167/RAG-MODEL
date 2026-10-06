import sys
from pathlib import Path

# Add rag-helpdesk/ to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import (
    load_documents,
    chunk_documents,
    print_chunk_statistics,
)


def main():
    print("\n=== LOADING PDF ===")

    documents = load_documents()

    print(
        f"Documents/pages loaded: {len(documents)}"
    )

    if not documents:
        print("\nERROR: No documents were loaded.")
        return

    print("\n=== CREATING CANONICAL CHUNKS ===")

    chunks = chunk_documents(documents)

    print(
        f"Total chunks created: {len(chunks)}"
    )

    if not chunks:
        print("\nERROR: No chunks were created.")
        return

    # Print chunk statistics
    print_chunk_statistics(chunks)

    print("\n=== FIRST 10 CHUNKS ===")

    for i, chunk in enumerate(chunks[:10]):
        print("\n" + "=" * 80)

        print(
            f"Chunk: {i}"
            f" | ID: {chunk.get('chunk_id')}"
            f" | Pages: {chunk.get('page_start')}-{chunk.get('page_end')}"
            f" | Tokens: {chunk.get('token_count')}"
            f" | Chapter: {chunk.get('chapter_num')}"
            f" | Section: {chunk.get('section_title')}"
        )

        print("-" * 80)

        print(
            chunk.get("text", "")[:700]
        )

    print("\n" + "=" * 80)
    print("CHUNKING TEST COMPLETE")


if __name__ == "__main__":
    main()