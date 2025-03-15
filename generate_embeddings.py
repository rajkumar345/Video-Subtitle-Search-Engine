from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch

def generate_embeddings(file_path, model_name='all-MiniLM-L6-v2'):
    # Load cleaned subtitle data
    df = pd.read_csv(file_path)
    df['chunks'] = df['chunks'].apply(eval)  # Convert string to list
    
    # Load Sentence Transformer Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name).to(device)
    
    embeddings = []
    for chunk_list in df['chunks']:
        chunk_embeddings = model.encode(chunk_list, convert_to_numpy=True, show_progress_bar=True)
        embeddings.append(chunk_embeddings)
    
    df['embeddings'] = embeddings
    df.to_pickle("subtitles_embeddings.pkl")
    return df

if __name__ == "__main__":
    df = generate_embeddings("subtitles_cleaned.csv")
    print("Embeddings generated and saved.")
