#!/usr/bin/env python3
"""
Migration script to move data from Chroma to PostgreSQL.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def migrate_chroma_to_postgres():
    """Migrate all chunks from Chroma to PostgreSQL."""
    print("=" * 60)
    print("CHROMA TO POSTGRESQL MIGRATION SCRIPT")
    print("=" * 60)
    
    # Check if PostgreSQL is enabled
    if os.getenv("USE_POSTGRES", "false").lower() != "true":
        print("❌ PostgreSQL is not enabled in .env")
        print("   Set USE_POSTGRES=true and configure DATABASE_URL")
        return False
    
    try:
        # Connect to Chroma
        from src.vector_store import ChromaVectorStore
        from src.config import VECTORSTORE_DIR
        
        print("Connecting to Chroma...")
        chroma = ChromaVectorStore(persist_directory=VECTORSTORE_DIR)
        chroma_count = chroma.get_collection_count()
        
        if chroma_count == 0:
            print("❌ No data in Chroma to migrate")
            return False
        
        print(f"Found {chroma_count} chunks in Chroma")
        
        # Connect to PostgreSQL
        from src.postgres_vector_store import PostgresVectorStore
        
        print("Connecting to PostgreSQL...")
        postgres = PostgresVectorStore()
        
        # Fetch all data from Chroma
        print("Fetching data from Chroma...")
        data = chroma.collection.get(include=["documents", "metadatas", "embeddings"])
        
        if not data.get("ids"):
            print("❌ No data retrieved from Chroma")
            return False
        
        # Process in batches to avoid memory issues
        batch_size = 100
        total_chunks = len(data["ids"])
        
        print(f"Migrating {total_chunks} chunks in batches of {batch_size}...")
        
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch_texts = data["documents"][i:end_idx]
            batch_embeddings = data["embeddings"][i:end_idx]
            batch_metadatas = data["metadatas"][i:end_idx]
            batch_ids = data["ids"][i:end_idx]
            
            # Migrate batch to PostgreSQL
            postgres.add_documents(
                texts=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
            
            print(f"  Migrated {end_idx}/{total_chunks} chunks")
        
        # Verify migration
        postgres_count = postgres.get_collection_count()
        print(f"\n✅ Migration complete!")
        print(f"   Chroma: {chroma_count} chunks")
        print(f"   PostgreSQL: {postgres_count} chunks")
        
        if postgres_count == chroma_count:
            print("   ✅ All chunks migrated successfully")
        else:
            print(f"   ⚠️  Count mismatch - please verify")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("This script migrates data from Chroma to PostgreSQL.")
    print("")
    print("PREREQUISITES:")
    print("  1. PostgreSQL server must be running")
    print("  2. Database 'rag_helpdesk' must exist")
    print("  3. pgvector extension must be enabled")
    print("  4. USE_POSTGRES=true in .env")
    print("  5. DATABASE_URL configured correctly")
    print("")
    
    response = input("Are you ready to migrate? (yes/no): ").strip().lower()
    if response != "yes":
        print("Migration cancelled.")
        return
    
    if migrate_chroma_to_postgres():
        print("\n✅ Migration completed successfully!")
        print("\nNEXT STEPS:")
        print("  1. Verify data in PostgreSQL")
        print("  2. Test queries work correctly")
        print("  3. Consider archiving Chroma data")
    else:
        print("\n❌ Migration failed. Please check the error above.")

if __name__ == "__main__":
    main()
