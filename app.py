import streamlit as st
import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from transcribe_audio import transcribe_audio
from search_audio import search_audio

# Set page title and layout
st.set_page_config(page_title="🎬 Video Subtitles Search Engine", layout="wide")

# Paths
BANNER_PATH = r"C:\Users\kotip\OneDrive\Documents\innomatics\video_subtitles_search_engine\banner_picinnomatics.jpg"
DB_PATH = r"C:\Users\kotip\OneDrive\Documents\innomatics\task8\data\eng_subtitles_database.db"

# Load models
tfidf_vectorizer = TfidfVectorizer()
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Display banner image
if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH, use_container_width=True)

# App title
st.title("🎬 Video Subtitles Search Engine")

# Sidebar for navigation
st.sidebar.title("🔍 Navigation")
option = st.sidebar.radio("Choose an option:", ["🎤 Transcribe Audio", "🔍 Search Audio", "📄 Search by Text"])

# ✅ **Function: Check if 'subtitles' table exists in the database**
def check_table_exists():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='subtitles';")
            return cursor.fetchone() is not None
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        return False

# ✅ **Function: Read subtitle data from database**
def read_subtitle_data():
    if not check_table_exists():
        st.error("❌ Table 'subtitles' does not exist in the database!")
        return pd.DataFrame()
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT id, filename, subtitle_text FROM subtitles", conn)
        
        if "subtitle_text" not in df.columns:
            st.error("❌ 'subtitle_text' column not found in the database!")
            return pd.DataFrame()
        
        return df
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return pd.DataFrame()

# ✅ **Function: Vectorize subtitles using TF-IDF**
def vectorize_tfidf(subtitles):
    if not subtitles.empty:
        return tfidf_vectorizer.fit_transform(subtitles)
    return None

# ✅ **Function: Vectorize subtitles using BERT**
def vectorize_bert(subtitles):
    if not subtitles.empty:
        return bert_model.encode(subtitles.tolist(), convert_to_tensor=True)
    return None

# ✅ **Function: Search subtitles using a text query**
def search_query(query, subtitles, tfidf_vectors, bert_embeddings):
    if tfidf_vectors is None or bert_embeddings is None:
        return None

    # Convert query to TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query])
    scores_tfidf = (query_vector @ tfidf_vectors.T).toarray().flatten()

    # Convert query to BERT embedding
    query_embedding = bert_model.encode(query, convert_to_tensor=True)
    scores_bert = util.pytorch_cos_sim(query_embedding, bert_embeddings)[0].cpu().numpy()

    # Combine scores using weighted sum
    combined_scores = (0.5 * scores_tfidf) + (0.5 * scores_bert)
    
    # Get top matches
    top_indices = np.argsort(combined_scores)[-5:][::-1]
    
    results = [
        {
            "filename": subtitles.iloc[idx]["filename"],
            "subtitle": subtitles.iloc[idx]["subtitle_text"],
            "score": combined_scores[idx]
        }
        for idx in top_indices
    ]
    return results

# 🎤 **Option: Transcribe Audio**
if option == "🎤 Transcribe Audio":
    st.header("🎤 Upload an Audio File for Transcription")
    uploaded_file = st.file_uploader("Upload your audio file:", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        st.text("Transcribing... ⏳")
        transcript = transcribe_audio("temp_audio.wav")
        st.text_area("📝 Transcribed Text:", transcript, height=150)

# 🔍 **Option: Search by Audio**
elif option == "🔍 Search Audio":
    st.header("🔍 Upload an Audio Query to Find Relevant Subtitles")
    search_file = st.file_uploader("Upload an audio query file:", type=["mp3", "wav", "m4a"])

    if search_file is not None:
        st.success("✅ Query file uploaded!")
        with open("query_audio.wav", "wb") as f:
            f.write(search_file.read())

        st.text("Searching for relevant subtitles... 🔎")
        results = search_audio("query_audio.wav")

        st.subheader("🎯 Most Relevant Subtitle Matches:")
        for idx, result in enumerate(results):
            st.write(f"📌 Match {idx + 1}: {result}")

# 📄 **Option: Search by Text**
elif option == "📄 Search by Text":
    st.subheader("🔍 Search Subtitles")
    query = st.text_input("Enter search query:")

    if query:
        subtitles = read_subtitle_data()
        
        if not subtitles.empty:
            # ✅ Vectorize subtitles correctly
            tfidf_vectors = vectorize_tfidf(subtitles["subtitle_text"])
            bert_embeddings = vectorize_bert(subtitles["subtitle_text"])
            
            if tfidf_vectors is not None and bert_embeddings is not None:
                results = search_query(query, subtitles, tfidf_vectors, bert_embeddings)
                
                if results:
                    st.write("### 🔥 Top Results:")
                    for result in results:
                        st.write(f"📂 **{result['filename']}**")
                        st.write(f"📝 {result['subtitle'][:300]}... (Score: {result['score']:.2f})")
                    st.success("✅ Search complete!")
                else:
                    st.error("❌ No matching results found!")
            else:
                st.error("❌ Error vectorizing subtitle text!")
        else:
            st.error("❌ Database is empty or missing required columns!")

# Footer
st.sidebar.write("---")
st.sidebar.write("🚀 Built with 😊 by K. Mrudula")
