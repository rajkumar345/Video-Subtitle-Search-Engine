import whisper
import torch
import chromadb
from sentence_transformers import SentenceTransformer

def search_audio(audio_path, collection_name="subtitles", top_k=5, model_size="small"):
    # Load Whisper Model
    whisper_model = whisper.load_model(model_size)
    
    # Transcribe Audio
    result = whisper_model.transcribe(audio_path)
    query = result['text']
    
    # Load Sentence Transformer Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    query_embedding = embed_model.encode(query, convert_to_numpy=True)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")

    # Check if the collection exists
    collection_names = client.list_collections()  # Returns a list of names
    if collection_name not in collection_names:
        print(f"Collection '{collection_name}' not found. Creating a new one...")
        collection = client.create_collection(name=collection_name)
    else:
        collection = client.get_collection(collection_name)

    # Retrieve results from ChromaDB
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

    return query, results.get('metadatas', [[]])[0]  # Handle empty results safely

if __name__ == "__main__":
    audio_file = "sample_audio.wav"
    query_text, retrieved_results = search_audio(audio_file)
    
    print("Query Text:", query_text)
    if retrieved_results:
        for res in retrieved_results:
            print(res)
    else:
        print("No matching results found in the database.")
