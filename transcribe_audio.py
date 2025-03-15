import whisper
import os

def transcribe_audio(audio_path, model_size="small"):
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found!")
        return None  # Explicitly return None if the file is missing
    
    # Load Whisper Model
    model = whisper.load_model(model_size)
    
    # Transcribe Audio
    result = model.transcribe(audio_path)
    
    return result['text']

if __name__ == "__main__":
    audio_file = "sample_audio.wav"
    transcript = transcribe_audio(audio_file)

    if transcript:
        print("Transcription:", transcript)
        
        # Save transcript to a text file
        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        
        print("Transcription saved to transcription.txt")
