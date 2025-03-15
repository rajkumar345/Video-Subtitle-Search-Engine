import chromadb
import pandas as pd
import numpy as np

def store_embeddings(file_path, collection_name="subtitles"):
    # Load embeddings
    df = pd.read_pickle(file_path)

    # Print column names for debugging
    print("🔍 Columns in DataFrame:", df.columns.tolist())

    # Ensure required columns exist
    required_columns = {'id', 'embeddings', 'chunks'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"❌ Missing required columns! Found: {df.columns}")

    # Check if 'file_name' exists, otherwise check alternatives
    file_column = None
    for col in ['file_name', 'filename', 'file']:
        if col in df.columns:
            file_column = col
            break

    if not file_column:
        raise ValueError("❌ Could not find 'file_name', 'filename', or 'file' column in DataFrame.")

    # Initialize ChromaDB Client
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(collection_name)

    # Store embeddings with metadata
    for idx, row in df.iterrows():
        for chunk_idx, emb in enumerate(row['embeddings']):
            collection.add(
                ids=[f"{row['id']}_{chunk_idx}"],
                embeddings=[emb.tolist() if isinstance(emb, np.ndarray) else emb],
                metadatas=[{"text": row['chunks'][chunk_idx], "file": row[file_column]}]
            )
    
    print("✅ Embeddings stored in ChromaDB.")

if __name__ == "__main__":
    store_embeddings("subtitles_embeddings.pkl")
