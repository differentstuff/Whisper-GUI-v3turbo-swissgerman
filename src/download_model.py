from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import src.print_color as pc


def download_model(model_id="nizarmichaud/whisper-large-v3-turbo-swissgerman"):
    """
    Downloads and caches the Whisper model and processor.
    """
    pc.print_info(f"\nDownloading model {model_id} (this may take a while)")

    try:
        # Set up cache directory
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "model_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        pc.print_info(f"Model will be cached in: {cache_dir}")

        # Download processor
        try:
            processor = WhisperProcessor.from_pretrained(
                model_id, cache_dir=cache_dir, local_files_only=False
            )
        except Exception as e:
            error_msg = f"Failed to download processor: {str(e)}"
            raise Exception(error_msg)

        # Download model
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id, cache_dir=cache_dir, local_files_only=False
            )
        except Exception as e:
            error_msg = f"Failed to download model: {str(e)}"
            raise Exception(error_msg)

        pc.print_success("Model downloaded successfully")

        return processor, model

    except Exception as e:
        error_msg = f"Error during model download: {str(e)}"
        pc.print_error(f"\nError: {error_msg}")
        raise


if __name__ == "__main__":
    download_model()
