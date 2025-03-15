import chromadb
from sentence_transformers import SentenceTransformer
import torch

def retrieve_text(query, collection_name="subtitles", top_k=5):
    # Load model for embedding queries
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    # Generate query embedding
    query_embedding = model.encode(query, convert_to_numpy=True)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(collection_name)
    
    # Retrieve results
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
    
    return results['metadatas'][0]

if __name__ == "__main__":
    query = "What is the movie about?"
    results = retrieve_text(query)
    for res in results:
        print(res)
