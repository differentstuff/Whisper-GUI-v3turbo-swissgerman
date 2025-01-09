import asyncio
import os
import torch
import psutil
from nicegui import app, ui
from src.model_handler import WhisperModelHandler
from src.gui_handler import AudioSplitter, TranscriptionHandler

# Constants
OUTPUT_FORMATS = ['xlsx', 'srt', 'txt', 'vtt', 'tsv', 'json']
FILEPICKER_FORMATS_AUDIO = "Audio Files (*.mp3;*.m4a;*.m4b;*.m4p;*.flac;*.ogg;*.oga;*.mogg;*.wav;*.wma;*.mmf;*.aa;*.aax)"
FILEPICKER_FORMATS_VIDEO = "Video Files (*.webm;*.mkv;*.flv;*.vob;*.ogv;*.ogg;*.drc;*.avi;*.mts;*.m2ts;*.ts;*.mov;*.qt;*.wmv;*.rm;*.rmvb;*.viv;*.asf;*.amv;*.mp4;*.m4p;*.m4v;*.mpg;*.mp2;*.mpeg;*.mpe;*.mpv;*.m2v;*.m4v;*.svi;*.3gp;*.3g2;*.f4v;*.f4p;*.f4a;*.f4b)"

async def choose_files(viewmodel):
    """Handle file selection dialog and update UI accordingly"""
    viewmodel.selected_files = await app.native.main_window.create_file_dialog(
        allow_multiple=True, 
        file_types=[FILEPICKER_FORMATS_AUDIO, FILEPICKER_FORMATS_VIDEO, "All Files (*)"]
    )
    
    if viewmodel.selected_files:
        need_splitting_count = sum(1 for file in viewmodel.selected_files 
                                 if AudioSplitter.get_segment_count(file) > 1)
        if need_splitting_count > 0:
            ui.notify(f'{"1 file needs" if need_splitting_count == 1 else f"{need_splitting_count} files need"} to be split')
    
    viewmodel.update_button_states()

def abort_transcription(viewmodel):
    """Handle transcription abort request"""
    viewmodel.abort_requested = True
    print("\nAborting transcription after current file completes...")
    ui.notify("Aborting transcription...")

async def transcribe_files(files, output_format, viewmodel, model, update_ui):
    """Transcribe audio files with comprehensive error handling and user feedback"""
    # Print debug info
    viewmodel.update_ui("\n=== Starting Transcription ===")
    viewmodel.update_ui(f"• Files: {len(files) if files else 0}")
    viewmodel.update_ui(f"• Formats: {output_format}")
    viewmodel.update_ui(f"• Model: {'Initialized' if model else 'Not initialized'}")
    viewmodel.update_ui(f"• Processor: {'Loaded' if model and model.processor else 'Not loaded'}")
    viewmodel.update_ui(f"• Device: {model.device if model else 'None'}")
    
    if not files:
        viewmodel.update_ui('No files selected', 'warning')
        return
        
    if not output_format:
        viewmodel.update_ui('No output format selected', 'warning')
        return
        
    if model is None:
        viewmodel.update_ui("Error: Model not initialized", 'negative')
        return
    
    # Set initial file count
    viewmodel.file_count = len(files)
    print(f"File count set to: {viewmodel.file_count}")

    # Initialize state
    viewmodel.abort_requested = False
    viewmodel.is_transcribing = True
    viewmodel.button_abort_visible = True
    viewmodel.button_run_enabled = False  # Disable start button while transcribing
    viewmodel.spinner_progress_visibility = True
    total_segments = 0
    processed_files = []
    print("\nStarting transcription...")
    
    try:
        viewmodel.update_ui("\nPreparing transcription...")
        
        # Validate files exist and are accessible
        for file in files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File not found: {file}")
            if not os.access(file, os.R_OK):
                raise PermissionError(f"Cannot read file: {file}")
        
        # Calculate total segments
        try:
            total_segments = sum(AudioSplitter.get_segment_count(file) for file in files)
            print(f"Total segments to process: {total_segments}")
            viewmodel.segment_count = total_segments
            viewmodel.update_label_progress()
        except Exception as e:
            raise RuntimeError(f"Error calculating segments: {str(e)}")
        
        # Verify model is initialized
        if model is None:
            viewmodel.update_ui("\nError: Model not initialized", 'negative')
            return
        
        # Load model if needed
        if model.processor is None:
            viewmodel.update_ui("\n=== Loading Model ===")
            viewmodel.update_ui("Loading Swiss German specialized model...")
            viewmodel.update_ui("This will download ~3GB of data on first run.")
            viewmodel.update_ui("Subsequent runs will use the cached model.")
            try:
                await asyncio.to_thread(model.load_model)
                viewmodel.update_ui("✓ Model loaded successfully")
                if model.device == "cuda":
                    viewmodel.update_ui(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                    viewmodel.update_ui(f"GPU: {torch.cuda.get_device_name(0)}")
                else:
                    viewmodel.update_ui(f"CPU Memory Usage: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")
            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}"
                viewmodel.update_ui(f"\n❌ {error_msg}", 'negative')
                # Reset UI state on error
                viewmodel.is_transcribing = False
                viewmodel.button_abort_visible = False
                viewmodel.button_run_enabled = True
                viewmodel.spinner_progress_visibility = False
                return
        
        # Process each file
        for file in files:
            if viewmodel.abort_requested:
                print("\nTranscription aborted by user.")
                break
            
            try:
                print(f"\nProcessing: {os.path.basename(file)}")
                file_segments = await asyncio.to_thread(AudioSplitter.split_audio, file)
                results = []
                
                for i, (segment_file, start_time) in enumerate(file_segments, 1):
                    print(f"Transcribing segment {i}/{len(file_segments)}...")
                    result = await asyncio.to_thread(model.transcribe, segment_file)
                    
                    # Adjust timestamps for segments
                    if start_time > 0:
                        for segment in result["segments"]:
                            segment["start"] += start_time / 1000
                            segment["end"] += start_time / 1000
                    
                    results.append(result)
                    viewmodel.segment_done_count += 1
                    viewmodel.update_label_progress()
                
                # Save results
                if len(results) > 1:
                    print("Combining segments...")
                    combined_result = results[0].copy()
                    combined_result["text"] = "\n".join(r["text"] for r in results)
                    combined_result["segments"] = []
                    for r in results:
                        combined_result["segments"].extend(r["segments"])
                    await asyncio.to_thread(TranscriptionHandler.save_result, combined_result, output_format, file)
                else:
                    await asyncio.to_thread(TranscriptionHandler.save_result, results[0], output_format, file)
                
                processed_files.append(file)
                print(f"Completed: {os.path.basename(file)}")
                
            except Exception as e:
                print(f"\nError processing file {os.path.basename(file)}: {str(e)}")
                viewmodel.update_ui(f"Error processing {os.path.basename(file)}", "negative")
                continue
            finally:
                # Cleanup temp files
                if len(file_segments) > 1:
                    await asyncio.to_thread(AudioSplitter.clear_temp_dir)
    
    except Exception as e:
        print(f"\nError during transcription: {str(e)}")
        viewmodel.update_ui(f"Error: {str(e)}", "negative")
    
    finally:
        # Reset UI state
        viewmodel.is_transcribing = False
        viewmodel.button_abort_visible = False
        viewmodel.spinner_progress_visibility = False
        viewmodel.update_button_states()  # Update button state based on current files/formats
        
        # Update progress
        viewmodel.file_count = 0
        viewmodel.segment_count = 0
        viewmodel.segment_done_count = 0
        viewmodel.update_label_progress()
        
        # Show final status
        if viewmodel.abort_requested:
            viewmodel.update_ui("Transcription aborted", "warning")
            viewmodel.update_ui(f"\nTranscription aborted. Processed {len(processed_files)} of {len(files)} files.")
        elif processed_files:
            viewmodel.update_ui("Transcription complete", "positive")
            viewmodel.update_ui(f"\nTranscription complete. Successfully processed {len(processed_files)} files.")
        else:
            viewmodel.update_ui("No files were processed", "warning")
