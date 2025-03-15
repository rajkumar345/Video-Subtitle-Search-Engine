import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from joblib import Memory

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

cachedir = ".cache"
memory = Memory(location=cachedir, verbose=0)

@memory.cache
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)  # Remove timestamps
        text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = ' '.join(char for char in text if ord(char) < 128)  # Remove non-ASCII
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
        return ' '.join(words)
    return None

@memory.cache
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start: start + chunk_size]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # 🔹 Debugging: Print column names before accessing 'file_content'
    print("Columns in CSV file:", df.columns)

    # 🔹 Ensure 'file_content' exists before applying functions
    if 'file_content' in df.columns:
        df['file_content'] = df['file_content'].apply(clean_text)
        df['chunks'] = df['file_content'].apply(lambda x: chunk_text(x, chunk_size=500, overlap=50))
        df.to_csv("subtitles_cleaned.csv", index=False)
        return df
    else:
        print("❌ Error: 'file_content' column is missing in the dataset.")
        return None

if __name__ == "__main__":
    df = preprocess_data("subtitles_raw.csv")
    print(df.head())
