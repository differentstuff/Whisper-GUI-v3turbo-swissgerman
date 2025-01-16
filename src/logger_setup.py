import logging
from logging.handlers import RotatingFileHandler
import os

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


def setup_logger(name, log_file, level=logging.INFO, console=False):
    """Function to setup as many loggers as you want"""

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create file handler
    fh = RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5,
        mode="a",  # append mode
    )
    fh.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(fh)

    # Add console handler if requested
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


# Setup individual loggers
# For main.py
main_logger = setup_logger("main", "main.log")

# For download_model.py
download_logger = setup_logger("download_model", "download_model.log")

# For file_handler.py
file_handler_logger = setup_logger("file_handler", "file_handler.log")

# For gui_handler.py
gui_logger = setup_logger("gui_handler", "gui_handler.log")

# For model_handler.py
model_logger = setup_logger("model_handler", "model_handler.log")

# For verify_ffmpeg.py
ffmpeg_logger = setup_logger("verify_ffmpeg", "verify_ffmpeg.log")
