# first line: 20
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
