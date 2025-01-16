from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import sys
from colorama import init, Fore, Style

# Initialize Colorama
init()


def print_header(text, color=Fore.CYAN):
    print(f"\n{color}=== {text} ==={Style.RESET_ALL}\n")


def print_info(text, color=Fore.WHITE):
    print(f"{color}{text}{Style.RESET_ALL}")


def print_success(text):
    print(f"{Fore.GREEN}> {text}{Style.RESET_ALL}")


def print_warning(text):
    print(f"{Fore.YELLOW}! {text}{Style.RESET_ALL}")


def print_error(text):
    print(f"{Fore.RED}X {text}{Style.RESET_ALL}")


def download_model(model_id="nizarmichaud/whisper-large-v3-turbo-swissgerman"):
    """
    Downloads and caches the Whisper model and processor.
    """
    print_info(f"\nDownloading model {model_id} (this may take a while)")

    try:
        # Set up cache directory
        cache_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(cache_dir, "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        print_info(f"Model will be cached in: {cache_dir}")

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

        print_success("Model downloaded successfully")

    except Exception as e:
        error_msg = f"Error during model download: {str(e)}"
        print_error(f"\nError: {error_msg}")
        raise


if __name__ == "__main__":
    print_header(f"Whisper model downloader")
    download_model()
