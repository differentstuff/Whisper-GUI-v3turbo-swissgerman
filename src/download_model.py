from transformers import WhisperProcessor, WhisperForConditionalGeneration

def download_model():
    print("\nDownloading Swiss German Whisper model (this may take a while)...")
    model_id = "nizarmichaud/whisper-large-v3-turbo-swissgerman"
    
    # Download both processor and model
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    print("Model downloaded successfully!")

if __name__ == "__main__":
    download_model()
