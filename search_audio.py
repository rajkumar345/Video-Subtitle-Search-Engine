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
    
    # Retrieve results from ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(collection_name)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
    
    return query, results['metadatas'][0]

if __name__ == "__main__":
    audio_file = "sample_audio.wav"
    query_text, retrieved_results = search_audio(audio_file)
    print("Query Text:", query_text)
    for res in retrieved_results:
        print(res)
