import sys
import shutil
import src.print_color as pc
import logging
from src.logger_setup import ffmpeg_logger as logger


def verify_ffmpeg():
    """Verify ffmpeg installation in user space"""
    logger.info("Starting ffmpeg verification")
    pc.print_info("\nVerifying ffmpeg installation")

    # Check if ffmpeg is accessible
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        success_msg = f"Found ffmpeg Path: {ffmpeg_path}"
        logger.info(success_msg)
        pc.print_success("Installation of ffmpeg found")
        pc.print_info(success_msg)
        return True

    # Log error and installation instructions
    logger.error("ffmpeg not found in system PATH")
    pc.print_error("ERROR: ffmpeg not found!")
    pc.print_warning("Please install ffmpeg:")

    # Platform-specific installation instructions
    if sys.platform == "win32":
        install_instructions = [
            "1. Open Command Prompt (not as Administrator)",
            "2. Run: winget install ffmpeg",
            "3. Close all Command Prompt windows",
            "4. Start the application again",
        ]
        logger.info("Providing Windows installation instructions")
        for instruction in install_instructions:
            pc.print_info(f"\n{instruction}")
            logger.debug(f"Installation instruction: {instruction}")
    elif sys.platform == "darwin":
        install_cmd = "brew install ffmpeg"
        logger.info("Providing macOS installation instructions")
        pc.print_info(f"\nRun: {install_cmd}")
        logger.debug(f"Installation instruction: {install_cmd}")
    else:
        install_cmd = "sudo apt install ffmpeg"
        logger.info("Providing Linux installation instructions")
        pc.print_info(f"\nRun: {install_cmd}")
        logger.debug(f"Installation instruction: {install_cmd}")

    return False


if __name__ == "__main__":
    try:
        logger.info("Running verify_ffmpeg.py as main script")
        if not verify_ffmpeg():
            logger.error("ffmpeg verification failed")
            sys.exit(1)
        logger.info("ffmpeg verification completed successfully")
    except Exception as e:
        logger.error(
            f"Unexpected error during ffmpeg verification: {str(e)}", exc_info=True
        )
        sys.exit(1)
