# region Imports

from src.model_handler import WhisperModelHandler
import whisperx
from typing import Dict, Any, Tuple, Optional
import os
from src.logger_setup import model_logger as logger
import src.print_color as pc
from src.audio_config import WHISPERX_LANGUAGE_CODES
import logging

# endregion Imports


# region Model Handler

class WhisperXModelHandler(WhisperModelHandler):
    """WhisperX model handler that extends the base WhisperModelHandler"""

    def __init__(self, device=None, model_id=None):
        """Initialize WhisperX model handler"""
        super().__init__(device, model_id)
        self.alignment_model = None
        self.alignment_language = None

    def load_alignment_model(self, language: str):
        """Load the alignment model for a specific language"""
        try:
            if language == "Auto":
                # Get language code from Whisper's detection
                language_code = self._get_whisper_language_code()
            else:
                language_code = self._normalize_language_code(language)

            if (self.alignment_model is None or language_code != self.alignment_language):
                logger.info(f"Loading WhisperX alignment model for {language_code}")
                self.alignment_model = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device
                )
                self.alignment_language = language_code
        except Exception as e:
            logger.error(f"Failed to load WhisperX alignment model: {str(e)}")
            return None

    def _validate_segments(self, segments: list) -> bool:
        """Validate segment structure and timing"""
        try:
            if not segments:
                return False

            previous_end = 0
            for segment in segments:
                # Check required fields
                if not all(k in segment for k in ["start", "end", "text"]):
                    logger.warning("Missing required fields in segment")
                    return False

                # Validate timing
                if not isinstance(segment["start"], (int, float)) or not isinstance(segment["end"], (int, float)):
                    logger.warning("Invalid timing format")
                    return False

                if segment["start"] > segment["end"]:
                    logger.warning("Invalid segment timing: start > end")
                    return False

                if segment["start"] < previous_end - 0.1:  # Allow small overlap
                    logger.warning("Invalid segment timing: overlapping segments")
                    return False

                previous_end = segment["end"]

                # Validate text
                if not isinstance(segment["text"], str) or not segment["text"].strip():
                    logger.warning("Empty or invalid text in segment")
                    return False

            return True
        except Exception as e:
            logger.error(f"Segment validation error: {str(e)}")
            return False
    
    def _get_whisper_language_code(self) -> str:
        """Get language code from Whisper's detection

        Returns:
            str: Two-letter language code
        """
        try:
            # If we have a detected language from Whisper
            if hasattr(self, 'detected_language'):
                lang = self.detected_language.lower()
                if lang in WHISPERX_LANGUAGE_CODES:
                    logger.info(f"Using detected language code: {WHISPERX_LANGUAGE_CODES[lang]}")
                    return WHISPERX_LANGUAGE_CODES[lang]

                # If it's already a two-letter code
                if len(lang) == 2:
                    logger.info(f"Using provided language code: {lang}")
                    return lang

            # Default to English if no language detected
            logger.warning("No language detected, defaulting to English (en)")
            return "en"

        except Exception as e:
            logger.error(f"Error getting language code: {str(e)}")
            logger.warning("Defaulting to English (en)")
            return "en"

    def _normalize_language_code(self, language: str) -> str:
        """Normalize language code to two-letter format

        Args:
            language: Language code or name

        Returns:
            str: Two-letter language code
        """
        try:
            # If already a two-letter code
            if len(language) == 2:
                return language.lower()

            normalized = language.lower().strip()
            if normalized in WHISPERX_LANGUAGE_CODES:
                return WHISPERX_LANGUAGE_CODES[normalized]

            # Get default language (first position in WHISPERX_LANGUAGE_CODES)
            default_lang = list(WHISPERX_LANGUAGE_CODES.values())[0]
            logger.warning(f"Unsupported language: {language}, defaulting to {default_lang}")
            return default_lang

        except Exception as e:
            logger.error(f"Error normalizing language code: {str(e)}")
            # Get default language (first position in WHISPERX_LANGUAGE_CODES)
            return list(WHISPERX_LANGUAGE_CODES.values())[0]

        
    async def transcribe_with_timestamps(
        self, 
        audio_path: str, 
        initial_transcript: str,
        language: str = "Auto"
    ) -> Tuple[Optional[list], Optional[str]]:
        """
        Enhance timestamps using WhisperX alignment model

        Args:
            audio_path: Path to audio file
            initial_transcript: Initial Whisper transcription
            language: Language code

        Returns:
            Tuple[list, str]: Enhanced segments and text
        """
        try:
            logger.info("Starting WhisperX timestamp enhancement")

            # Validate inputs
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            if not initial_transcript:
                raise ValueError("Empty initial transcript")

            pc.print_info("\nEnhancing timestamps with WhisperX")

            # Load audio using WhisperX's loader
            try:
                audio = whisperx.load_audio(audio_path)
            except Exception as e:
                logger.error(f"Failed to load audio for WhisperX: {str(e)}")
                return None, initial_transcript
            
            # Load alignment model if needed
            alignment_model = self.load_alignment_model(language)
            if alignment_model is None:
                logger.warning("Failed to load alignment model, falling back to regular timestamps")
                return None, initial_transcript

            # Align with timeout protection
            try:
                result_aligned = whisperx.align(
                    initial_transcript,
                    self.alignment_model,
                    audio,
                    self.device,
                    return_char_alignments=False
                )
            except Exception as e:
                logger.error(f"Alignment failed: {str(e)}")
                return None, initial_transcript

            if result_aligned and "segments" in result_aligned and result_aligned["segments"]:
                # Validate segment structure
                if self._validate_segments(result_aligned["segments"]):
                    logger.info("WhisperX alignment successful")
                    return result_aligned["segments"], initial_transcript
                else:
                    logger.warning("Invalid segment structure")
                    return None, initial_transcript
            else:
                logger.warning("WhisperX alignment produced no valid segments")
                return None, initial_transcript

        except Exception as e:
            logger.error(f"WhisperX timestamp enhancement failed: {str(e)}")
            pc.print_warning(f"WhisperX enhancement failed: {str(e)}")
            pc.print_info("Falling back to regular timestamps")
            return None, initial_transcript

    def _prepare_whisperx_config(self) -> Dict[str, Any]:
        """Prepare WhisperX-specific configuration"""
        return {
            "align_model": {
                "use_auto_language": True,
                "interpolate_method": "linear",
            },
            "return_char_alignments": False,
            "vad_onset": 0.500,
            "vad_offset": 0.363
        }

# endregion Model Handler