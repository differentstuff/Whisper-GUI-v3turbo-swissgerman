from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

def download_model():
    print("\nDownloading Swiss German Whisper model (this may take a while)...")
    model_id = "nizarmichaud/whisper-large-v3-turbo-swissgerman"
    
    # Set up cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Model will be cached in: {cache_dir}")
    
    # Download both processor and model
    processor = WhisperProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=False  # Allow download if not present
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=False,  # Allow download if not present
        use_legacy_cache=False   # Avoid deprecation warning
    )
    
    print("Model downloaded successfully!")

if __name__ == "__main__":
    download_model()
