from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

def download_model(model_id = "nizarmichaud/whisper-large-v3-turbo-swissgerman"):
    print(f"\nDownloading model {model_id} (this may take a while)...")
    
    # Set up cache directory
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache") # use os.path.dirname twice to go one level up instead of using '..'
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
        local_files_only=False  # Allow download if not present
    )
    
    print("Model downloaded successfully!")

if __name__ == "__main__":
    download_model()
