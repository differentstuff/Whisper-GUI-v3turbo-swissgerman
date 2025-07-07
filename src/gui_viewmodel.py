# region Imports

from dataclasses import dataclass
from typing import Optional, List, Any
from src.audio_config import TIMESTAMPS_CONFIG
from src.file_handler import transcribe_files

from src.logger_setup import gui_logger as logger

# endregion Imports


#region Dataclasses

@dataclass
class TranscriptionState:
    selected_files: Optional[List[str]] = None
    selected_output_formats: Optional[List[str]] = None
    is_transcribing: bool = False
    abort_requested: bool = False
    model: Any = None

    @property
    def file_count(self):
        return self._file_count

    @file_count.setter
    def file_count(self, value: int):
        self._file_count = value

    @property
    def file_count_old(self):
        return self._file_count_old

    @file_count_old.setter
    def file_count_old(self, value: int):
        self._file_count_old = value

    @property
    def segment_count(self):
        return self._segment_count

    @segment_count.setter
    def segment_count(self, value: int):
        self._segment_count = value

    @property
    def segment_done_count(self):
        return self._segment_done_count

    @segment_done_count.setter
    def segment_done_count(self, value: int):
        self._segment_done_count = value

@dataclass
class ProgressState:
    file_count: int = 0
    file_count_old: int = 0
    segment_count: int = 0
    segment_done_count: int = 0
    segment_current_progress: int = 0


#endregion Dataclasses


# region View Model

class ViewModel:
    """
    ViewModel handles all business logic and state management.
    """
    def __init__(self):
        logger.info("Initializing ViewModel")

        # Transcription State
        self._transcription = TranscriptionState()

        # Progress State
        self._progress = ProgressState()

        self._init_timestamp_config()

    def _init_timestamp_config(self):
        """Initialize timestamp configuration"""
        if "use_timestamps" not in TIMESTAMPS_CONFIG:
            TIMESTAMPS_CONFIG["use_timestamps"] = False
        if "use_whisperx" not in TIMESTAMPS_CONFIG:
            TIMESTAMPS_CONFIG["use_whisperx"] = True  # Default to WhisperX
    
    @property
    def selected_files(self):
        return self._transcription.selected_files

    @selected_files.setter
    def selected_files(self, value: List[str]):
        self._transcription.selected_files = value

    @property
    def is_transcribing(self):
        return self._transcription.is_transcribing

    @is_transcribing.setter
    def is_transcribing(self, value: bool):
        self._transcription.is_transcribing = value

    def update_output_formats(self, formats: List[str]):
        """Update output formats"""
        logger.info(f"Updating output formats: {formats}")
        self._transcription.selected_output_formats = formats

    def update_model_selection(self, model: str):
        """Update selected model"""
        logger.info(f"Updating model selection: {model}")
        self._transcription.model = model

    def update_language_selection(self, language: str):
        """Update selected language"""
        logger.info(f"Updating language selection: {language}")
        self._transcription.language = language

    def get_progress(self):
        """Get current progress"""
        return {
            "file_count": self._progress.file_count,
            "segment_count": self._progress.segment_count,
            "segment_done_count": self._progress.segment_done_count,
        }

    def update_progress(self, current: int, total: int):
        """Update progress state"""
        self._progress.segment_done_count = current
        self._progress.segment_count = total

    async def start_transcription(self):
        """Start transcription process"""
        if not self._validate_transcription_params():
            return False

        self._transcription.is_transcribing = True
        try:
            result = await self._run_transcription()
            return result
        finally:
            self._transcription.is_transcribing = False

    def _validate_transcription_params(self):
        """Validate transcription parameters"""
        if not self._transcription.selected_files:
            logger.warning("No files selected")
            return False
        if not self._transcription.selected_output_formats:
            logger.warning("No output format selected")
            return False
        return True

    async def _run_transcription(self):
        """Run the transcription process"""
        try:
            result = await transcribe_files(
                files=self._transcription.selected_files,
                output_format=self._transcription.selected_output_formats,
                viewmodel=self,
                model=self._transcription.model,
                model_name=self._transcription.model,
                language=self._transcription.language,
            )

            if result is not None:
                model, success = result
                if success:
                    self._transcription.model = model
                    logger.info("Transcription completed successfully")
                else:
                    logger.warning("Transcription completed with some errors")
                return success
            return False
        except Exception as e:
            self._handle_transcription_error(e)
            return False

    def _handle_transcription_error(self, error):
        """Handle transcription errors"""
        if isinstance(error, FileNotFoundError):
            logger.error(f"File not found: {str(error)}", exc_info=True)
        elif isinstance(error, PermissionError):
            logger.error(f"Permission error: {str(error)}", exc_info=True)
        else:
            logger.error(f"Unexpected error: {str(error)}", exc_info=True)

    def reset_progress(self):
        """Reset progress state"""
        self._progress.file_count = 0
        self._progress.segment_count = 0
        self._progress.segment_done_count = 0

    def abort_transcription(self):
        """Abort ongoing transcription"""
        logger.info("Transcription abort requested")
        self._transcription.abort_requested = True

    def is_abort_requested(self):
        """Check if abort was requested"""
        return self._transcription.abort_requested

# endregion View Model
