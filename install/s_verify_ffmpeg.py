import sys
import shutil
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


def verify_ffmpeg():
    """Verify ffmpeg installation in user space"""
    # Check if ffmpeg is accessible
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print_success("Found ffmpeg")
        print_info(f"Found fmpeg Path: {ffmpeg_path}")
        return True

    # Log error and installation instructions
    print_error("ERROR: ffmpeg not found!")
    print_warning("Please install ffmpeg:")

    # Platform-specific installation instructions
    if sys.platform == "win32":
        install_instructions = [
            "1. Open Command Prompt (not as Administrator)",
            "2. Run: winget install ffmpeg",
            "3. Close all Command Prompt windows",
            "4. Start the application again",
        ]
        for instruction in install_instructions:
            print_info(f"\n{instruction}")
    elif sys.platform == "darwin":
        install_cmd = "brew install ffmpeg"
        logger.info("Providing macOS installation instructions")
        print_info(f"\nRun: {install_cmd}")
        logger.debug(f"Installation instruction: {install_cmd}")
    else:
        install_cmd = "sudo apt install ffmpeg"
        logger.info("Providing Linux installation instructions")
        print_info(f"\nRun: {install_cmd}")
        logger.debug(f"Installation instruction: {install_cmd}")

    return False


if __name__ == "__main__":
    try:
        print_header("Verifying ffmpeg installation")
        if not verify_ffmpeg():
            sys.exit(1)
    except Exception as e:

        sys.exit(1)
