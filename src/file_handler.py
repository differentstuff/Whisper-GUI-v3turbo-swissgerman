# region Imports

import asyncio
import os
import torch
import psutil
from typing import Any, List, Tuple, Optional
from nicegui import app, ui
from src.model_handler import WhisperModelHandler
from src.model_handler_x import WhisperXModelHandler
from src.audio_handler import AudioSplitter, TranscriptionHandler
import src.print_color as pc
from src.logger_setup import file_handler_logger as logger
from src.audio_config import TIMESTAMPS_CONFIG, AUDIO_CONFIG, FILEPICKER_FORMATS_VIDEO, FILEPICKER_FORMATS_AUDIO

# endregion Imports


# region File Selection
async def choose_files(viewmodel):
    """Handle file selection dialog and update UI accordingly"""
    logger.info("Opening file selection dialog")

    viewmodel.selected_files = await app.native.main_window.create_file_dialog(
        allow_multiple=True,
        file_types=[
            FILEPICKER_FORMATS_AUDIO,
            FILEPICKER_FORMATS_VIDEO,
            "All Files (*)",
        ],
    )

    if viewmodel.selected_files:
        logger.info(f"Selected {len(viewmodel.selected_files)} files")
        with AudioSplitter() as splitter:
            need_splitting_count = sum(1 for file in viewmodel.selected_files
                if splitter.get_segment_count(file) > 1
            )
        if need_splitting_count > 0:
            ui.notify(
                f'{"1 file needs" if need_splitting_count == 1 else f"{need_splitting_count} files need"} to be split'
            )

    viewmodel.update_button_states()

# endregion File Selection


# region Control Functions
def abort_transcription(viewmodel):
    """Handle transcription abort request"""
    logger.info("Transcription abort requested")
    viewmodel.abort_requested = True
    pc.print_warning("\nAborting transcription after current file completes")
    ui.notify("Aborting transcription")

# endregion Control Functions


# region Transcription

async def _initialize_whisperx(model: WhisperModelHandler) -> Tuple[Optional[WhisperXModelHandler], bool]:
    """Initialize WhisperX handler if enabled"""
    use_whisperx = TIMESTAMPS_CONFIG.get("use_whisperx", True)
    
    if not use_whisperx:
        return None, False

    try:
        if model.device == "cuda":
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            if free_memory < AUDIO_CONFIG["model"]["min_gpu_memory"] * 1024 * 1024:  # 2GB minimum
                logger.warning("Insufficient GPU memory for WhisperX")
                pc.print_warning("Insufficient GPU memory for WhisperX, falling back to regular timestamps")
                return None, False
            
        whisperx_handler = WhisperXModelHandler(device=model.device)
        logger.info("WhisperX initialized successfully")
        pc.print_info("WhisperX initialized for timestamp enhancement")
        return whisperx_handler, True
    
    except ImportError as ie:
        logger.warning(f"WhisperX not installed: {str(ie)}")
        pc.print_warning("WhisperX not installed, falling back to regular timestamps")
    except RuntimeError as re:
        logger.warning(f"WhisperX runtime error: {str(re)}")
        pc.print_warning("WhisperX initialization failed, falling back to regular timestamps")
    except Exception as e:
        logger.warning(f"Failed to initialize WhisperX: {str(e)}")
        pc.print_warning("Falling back to regular timestamps")
    
    return None, False


async def _enhance_timestamps(
    whisperx_handler: WhisperXModelHandler,
    segment_file: str,
    result: dict,
    language: str
) -> dict:
    """Enhance timestamps using WhisperX"""
    try:
        enhanced_result = await asyncio.to_thread(
            whisperx_handler.transcribe_with_timestamps,
            segment_file,
            result['text'],
            language
        )
        if enhanced_result and enhanced_result[0]:
            logger.info("WhisperX successfully enhanced timestamps")
            result['segments'] = enhanced_result[0]
            result['text'] = enhanced_result[1]
        else:
            logger.warning("WhisperX enhancement failed, using original timestamps")
    except Exception as wx_e:
        logger.error(f"WhisperX enhancement failed: {str(wx_e)}")
        pc.print_warning("Falling back to original timestamps")

    return result


async def transcribe_files(
    self,
    files: List[str],
    output_format: List[str],
    viewmodel,
    model: WhisperModelHandler,
    model_name: str,
    language: str = "Auto",
    **kwargs: Any
) -> Optional[Tuple[WhisperModelHandler, bool]]:
    """
    Transcribe audio files with support for custom model and language selection

    Args:
        files: List of audio files to transcribe
        output_format: List of output formats
        viewmodel: ViewModel instance
        model: WhisperModelHandler instance
        update_ui: UI update callback
        model_name: Name of the model to use
        language: Language for transcription (default: Auto)
        **kwargs: Additional arguments

        Returns:
            Optional[Tuple[WhisperModelHandler, bool]]: Updated model instance and success flag
    """
    logger.info(
        f"Starting transcription with {len(files) if files else 0} files, model: {model_name}, language: {language}"
    )

    # region Initial Setup
    viewmodel.update_ui("\n=== Starting Transcription ===")
    viewmodel.update_ui(f"• Files: {len(files) if files else 0}")
    viewmodel.update_ui(f"• Formats: {output_format}")
    viewmodel.update_ui(f"• Model: {'Initialized' if model else 'Not initialized'}")
    viewmodel.update_ui(f"• Language: {language}")
    viewmodel.update_ui(
        f"• Processor: {'Loaded' if model and model.processor else 'Not loaded'}"
    )
    viewmodel.update_ui(f"• Device: {model.device if model else 'None'}")

    if files:
        need_splitting_count = sum(
            1
            for file in viewmodel.selected_files
            if AudioSplitter.get_segment_count(file) > 1
        )
        if need_splitting_count > 0:
            logger.info(f"{need_splitting_count} files need splitting")
            ui.notify(
                f'{"1 file needs" if need_splitting_count == 1 else f"{need_splitting_count} files need"} to be split'
            )

    if not files:
        logger.warning("No files selected")
        viewmodel.update_ui("No files selected", "warning")
        return

    if not output_format:
        logger.warning("No output format selected")
        viewmodel.update_ui("No output format selected", "warning")
        return

    if model is None:
        logger.info("Model not initialized, initializing now")
        viewmodel.update_ui("\nInitializing model for first use...")
        try:
            model = WhisperModelHandler(
                device="cuda" if torch.cuda.is_available() else "cpu",
                model_id=model_name,
            )
            await asyncio.to_thread(model.load_model)
            viewmodel.model = model  # Store the model in viewmodel
            logger.info(f"Model initialized successfully: {model_name}")
            viewmodel.update_ui(f"> Model loaded successfully: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            viewmodel.update_ui(f"\nError: Failed to initialize model: {str(e)}", "negative")
            raise
    # endregion Initial Setup

    # region State Initialization
    viewmodel.file_count = len(files)
    pc.print_info(f"File count set to: {viewmodel.file_count}")
    logger.info(f"File count set to: {viewmodel.file_count}")

    viewmodel.abort_requested = False
    viewmodel.is_transcribing = True
    viewmodel.button_abort_visible = True
    viewmodel.button_run_enabled = False
    viewmodel.spinner_progress_visibility = True
    total_segments = 0
    processed_files = []
    pc.print_info("\nStarting transcription")
    # endregion State Initialization

    try:
        # region Preparation
        viewmodel.update_ui("\nPreparing transcription")

        for file in files:
            if not os.path.exists(file):
                logger.error(f"File not found: {file}")
                raise FileNotFoundError(f"File not found: {file}")
            if not os.access(file, os.R_OK):
                logger.error(f"Cannot read file: {file}")
                raise PermissionError(f"Cannot read file: {file}")

        try:
            with AudioSplitter() as splitter:
                total_segments = sum(
                    splitter.get_segment_count(file) for file in files
                )
            pc.print_info(f"Total segments to process: {total_segments}")
            logger.info(f"Total segments to process: {total_segments}")
            viewmodel.segment_count = total_segments
            viewmodel.update_label_progress()
        except Exception as e:
            logger.error(f"Error calculating segments: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error calculating segments: {str(e)}")
        # endregion Preparation

        # region Model Setup
        if model is None:
            logger.error("Model not initialized")
            viewmodel.update_ui("\nError: Model not initialized", "negative")
            return

        need_new_model = model is None or (model_name and model.model_id != model_name)
        if need_new_model:
            logger.info(f"Loading new model: {model_name}")
            viewmodel.update_ui("\n=== Loading Model ===")
            viewmodel.update_ui(f"Loading model: {model_name}")
            viewmodel.update_ui("This will download model data on first run.")
            viewmodel.update_ui("Subsequent runs will use the cached model.")
            try:
                model = WhisperModelHandler(
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    model_id=model_name,
                )
                await asyncio.to_thread(model.load_model)
                viewmodel.update_ui(f"> Model loaded successfully: {model_name}")
                if model.device == "cuda":
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"GPU Memory Usage: {gpu_memory:.1f}MB")
                    viewmodel.update_ui(f"GPU Memory Usage: {gpu_memory:.1f}MB")
                    viewmodel.update_ui(f"GPU: {torch.cuda.get_device_name(0)}")
                else:
                    cpu_memory = psutil.Process().memory_info().rss / 1024**2
                    logger.info(f"CPU Memory Usage: {cpu_memory:.1f}MB")
                    viewmodel.update_ui(f"CPU Memory Usage: {cpu_memory:.1f}MB")
                viewmodel.model = model
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}", exc_info=True)
                error_msg = f"Failed to load model: {str(e)}"
                viewmodel.update_ui(f"\nX {error_msg}", "negative")
                viewmodel.is_transcribing = False
                viewmodel.button_abort_visible = False
                viewmodel.button_run_enabled = True
                viewmodel.spinner_progress_visibility = False
                return
        # endregion Model Setup

        # region File Processing
        for file in files:
            if viewmodel.abort_requested:
                logger.info("Transcription aborted by user")
                pc.print_warning("\nTranscription aborted by user.")
                break

            try:
                logger.info(f"Processing file: {os.path.basename(file)}")
                pc.print_info(f"\nProcessing: {os.path.basename(file)}")
                with AudioSplitter() as splitter:
                    file_segments = await asyncio.to_thread(splitter.split_audio, file)
                results = []

                # Initialize WhisperX only if both timestamps and WhisperX are enabled
                if TIMESTAMPS_CONFIG.get("use_whisperx", True):
                    whisperx_handler, use_whisperx = await self._initialize_whisperx(viewmodel.model)
                else:
                    whisperx_handler, use_whisperx = None, False

                for i, (segment_file, start_time) in enumerate(file_segments, 1):
                    logger.info(f"Transcribing segment {i}/{len(file_segments)}")
                    pc.print_info(f"Transcribing segment {i}/{len(file_segments)}")

                    # Initial transcription with Whisper
                    result = await asyncio.to_thread(
                        model.transcribe,
                        segment_file,
                        language=language,
                        timestamps_enabled=not use_whisperx,
                    )

                    # Enhance timestamps with WhisperX if enabled
                    if use_whisperx and result:
                        result = await self._enhance_timestamps(
                            whisperx_handler,
                            segment_file,
                            result,
                            language
                        )

                    # Adjust timestamps for split files
                    if start_time > 0:
                        for segment in result["segments"]:
                            segment["start"] += start_time / 1000
                            segment["end"] += start_time / 1000

                    results.append(result)
                    viewmodel.segment_done_count += 1
                    viewmodel.update_label_progress()

                # Combine results and save
                if len(results) > 1:
                    logger.info("Combining segments")
                    pc.print_info("Combining segments")
                    combined_result = results[0].copy()
                    combined_result["text"] = "\n".join(r["text"] for r in results)
                    combined_result["segments"] = []
                    for r in results:
                        combined_result["segments"].extend(r["segments"])
                    await asyncio.to_thread(
                        TranscriptionHandler.save_result,
                        combined_result,
                        output_format,
                        file,
                    )
                else:
                    await asyncio.to_thread(
                        TranscriptionHandler.save_result,
                        results[0],
                        output_format,
                        file,
                    )

                processed_files.append(file)
                logger.info(f"Successfully processed file: {os.path.basename(file)}")
                pc.print_success(
                    f"Transcription completed of file: {os.path.basename(file)}"
                )

            except Exception as e:
                logger.error(
                    f"Error processing file {os.path.basename(file)}: {str(e)}",
                    exc_info=True,
                )
                pc.print_error(
                    f"\nError processing file {os.path.basename(file)}: {str(e)}"
                )
                viewmodel.update_ui(
                    f"Error processing {os.path.basename(file)}", "negative"
                )
                continue
            finally:
                if whisperx_handler:
                    whisperx_handler._cleanup()
        # endregion File Processing

    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        pc.print_error(f"\nError during transcription: {str(e)}")
        viewmodel.update_ui(f"Error: {str(e)}", "negative")

    finally:
        # region Cleanup
        viewmodel.is_transcribing = False
        viewmodel.button_abort_visible = False
        viewmodel.spinner_progress_visibility = False
        viewmodel.update_button_states()

        viewmodel.file_count = 0
        viewmodel.segment_count = 0
        viewmodel.segment_done_count = 0
        viewmodel.update_label_progress()

        # Clean up WhisperX resources if they exist
        if 'whisperx_handler' in locals() and whisperx_handler:
            whisperx_handler._cleanup()
            
        if viewmodel.abort_requested:
            logger.info(
                f"Transcription aborted. Processed {len(processed_files)} of {len(files)} files"
            )
            viewmodel.update_ui("Transcription aborted", "warning")
            viewmodel.update_ui(
                f"\nTranscription aborted. Processed {len(processed_files)} of {len(files)} files."
            )
        elif processed_files:
            logger.info(
                f"Transcription complete. Successfully processed {len(processed_files)} files"
            )
            viewmodel.update_ui("Transcription complete", "positive")
            viewmodel.update_ui(
                f"\nTranscription complete. Successfully processed {len(processed_files)} files."
            )
        else:
            logger.warning("No files were processed")
            viewmodel.update_ui("No files were processed", "warning")
        # endregion Cleanup
        
    pc.print_info("Transcription ended")

# endregion Transcription
