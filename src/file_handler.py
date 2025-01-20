# region Imports
import asyncio
import os
import torch
import psutil
from nicegui import app, ui
from src.model_handler import WhisperModelHandler
from src.gui_handler import AudioSplitter, TranscriptionHandler
import src.print_color as pc
from src.logger_setup import file_handler_logger as logger

# endregion Imports


# region Constants
OUTPUT_FORMATS = ["xlsx", "srt", "txt", "vtt", "tsv", "json"]
FILEPICKER_FORMATS_AUDIO = "Audio Files (*.mp3;*.m4a;*.m4b;*.m4p;*.flac;*.ogg;*.oga;*.mogg;*.wav;*.wma;*.mmf;*.aa;*.aax)"
FILEPICKER_FORMATS_VIDEO = "Video Files (*.webm;*.mkv;*.flv;*.vob;*.ogv;*.ogg;*.drc;*.avi;*.mts;*.m2ts;*.ts;*.mov;*.qt;*.wmv;*.rm;*.rmvb;*.viv;*.asf;*.amv;*.mp4;*.m4p;*.m4v;*.mpg;*.mp2;*.mpeg;*.mpe;*.mpv;*.m2v;*.m4v;*.svi;*.3gp;*.3g2;*.f4v;*.f4p;*.f4a;*.f4b)"
# endregion Constants


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
        need_splitting_count = sum( 1 for file in viewmodel.selected_files
            if AudioSplitter.get_segment_count(file) > 1
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
async def transcribe_files(
    files,
    output_format,
    viewmodel,
    model,
    update_ui,
    model_name,
    language="Auto",
    timestamps_enabled=False,
):
    """
    Transcribe audio files with support for custom model and language selection

    :param files: List of audio files to transcribe
    :param output_format: List of output formats
    :param viewmodel: ViewModel instance
    :param model: WhisperModelHandler instance
    :param update_ui: UI update callback
    :param model_name: Name of the model to use (default: Swiss German model)
    :param language: Language for transcription (default: Auto)
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
        logger.error("Model not initialized")
        viewmodel.update_ui("Error: Model not initialized", "negative")
        return
    # endregion

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
    # endregion

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
            total_segments = sum(
                AudioSplitter.get_segment_count(file) for file in files
            )
            pc.print_info(f"Total segments to process: {total_segments}")
            logger.info(f"Total segments to process: {total_segments}")
            viewmodel.segment_count = total_segments
            viewmodel.update_label_progress()
        except Exception as e:
            logger.error(f"Error calculating segments: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error calculating segments: {str(e)}")
        # endregion

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
        # endregion

        # region File Processing
        for file in files:
            if viewmodel.abort_requested:
                logger.info("Transcription aborted by user")
                pc.print_warning("\nTranscription aborted by user.")
                break

            try:
                logger.info(f"Processing file: {os.path.basename(file)}")
                pc.print_info(f"\nProcessing: {os.path.basename(file)}")
                file_segments = await asyncio.to_thread(AudioSplitter.split_audio, file)
                results = []

                for i, (segment_file, start_time) in enumerate(file_segments, 1):
                    logger.info(f"Transcribing segment {i}/{len(file_segments)}")
                    pc.print_info(f"Transcribing segment {i}/{len(file_segments)}")
                    result = await asyncio.to_thread(
                        model.transcribe,
                        segment_file,
                        language=language,
                        timestamps_enabled=timestamps_enabled,
                    )

                    if start_time > 0:
                        for segment in result["segments"]:
                            segment["start"] += start_time / 1000
                            segment["end"] += start_time / 1000

                    results.append(result)
                    viewmodel.segment_done_count += 1
                    viewmodel.update_label_progress()

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
                if len(file_segments) > 1:
                    await asyncio.to_thread(AudioSplitter.clear_temp_dir)
        # endregion

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
        # endregion


# endregion Transcription
