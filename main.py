import asyncio
import random
import os
import torch
import psutil
from nicegui import app, ui
from src.model_handler import WhisperModelHandler
from src.gui_handler import ViewModel, AudioSplitter, TranscriptionHandler, start_reading_console

# Suppress console output from NiceGUI
import logging
logging.getLogger('nicegui').setLevel(logging.WARNING)
logging.getLogger('watchfiles').setLevel(logging.WARNING)

# Suppress unnecessary output
os.environ['PYTHONWARNINGS'] = 'ignore'

# Constants
OUTPUT_FORMATS = ['xlsx', 'srt', 'txt', 'vtt', 'tsv', 'json']
FILEPICKER_FORMATS_AUDIO = "Audio Files (*.mp3;*.m4a;*.m4b;*.m4p;*.flac;*.ogg;*.oga;*.mogg;*.wav;*.wma;*.mmf;*.aa;*.aax)"
FILEPICKER_FORMATS_VIDEO = "Video Files (*.webm;*.mkv;*.flv;*.vob;*.ogv;*.ogg;*.drc;*.avi;*.mts;*.m2ts;*.ts;*.mov;*.qt;*.wmv;*.rm;*.rmvb;*.viv;*.asf;*.amv;*.mp4;*.m4p;*.m4v;*.mpg;*.mp2;*.mpeg;*.mpe;*.mpv;*.m2v;*.m4v;*.svi;*.3gp;*.3g2;*.f4v;*.f4p;*.f4a;*.f4b)"

viewmodel = ViewModel()
model = WhisperModelHandler()

async def choose_files():
    global viewmodel
    viewmodel.selected_files = await app.native.main_window.create_file_dialog(
        allow_multiple=True, 
        file_types=[FILEPICKER_FORMATS_AUDIO, FILEPICKER_FORMATS_VIDEO, "All Files (*)"]
    )
    
    if viewmodel.selected_files:
        need_splitting_count = sum(1 for file in viewmodel.selected_files 
                                 if AudioSplitter.get_segment_count(file) > 1)
        if need_splitting_count > 0:
            ui.notify(f'{"1 file needs" if need_splitting_count == 1 else f"{need_splitting_count} files need"} to be split')
    
    viewmodel.update_buttons()

def abort_transcription():
    global viewmodel
    viewmodel.abort_requested = True
    print("\nAborting transcription after current file completes...")
    ui.notify("Aborting transcription...")

async def transcribe_files(files, output_format):
    """Transcribe audio files with comprehensive error handling and user feedback"""
    if not files:
        ui.notify('No files selected', type='warning')
        return
        
    if not output_format:
        ui.notify('No output format selected', type='warning')
        return
    
    # Initialize state
    global viewmodel, model
    viewmodel.abort_requested = False
    viewmodel.is_transcribing = True
    viewmodel.button_abort_visible = True
    viewmodel.button_run_enabled = False
    viewmodel.spinner_progress_visibility = True
    total_segments = 0
    processed_files = []
    
    try:
        # Auto-expand console
        if viewmodel.console_expansion:
            viewmodel.console_expansion.expand()
            await asyncio.sleep(0.1)
        
        print("\nPreparing transcription...")
        
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
        
        # Load model if needed
        if model.processor is None:
            print("\nLoading Swiss German specialized model...")
            print("This will download ~3GB of data on first run.")
            print("Subsequent runs will use the cached model.")
            try:
                model.load_model()
                print("Model loaded successfully")
                if model.device == "cuda":
                    print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                else:
                    print(f"RAM usage: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")
        
        # Process each file
        for file in files:
            if viewmodel.abort_requested:
                print("\nTranscription aborted by user.")
                break
            
            try:
                print(f"\nProcessing: {os.path.basename(file)}")
                file_segments = AudioSplitter.split_audio(file)
                results = []
                
                for i, (segment_file, start_time) in enumerate(file_segments, 1):
                    print(f"Transcribing segment {i}/{len(file_segments)}...")
                    result = model.transcribe(segment_file)
                    
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
                    TranscriptionHandler.save_result(combined_result, output_format, file)
                else:
                    TranscriptionHandler.save_result(results[0], output_format, file)
                
                processed_files.append(file)
                print(f"Completed: {os.path.basename(file)}")
                
            except Exception as e:
                print(f"\nError processing file {os.path.basename(file)}: {str(e)}")
                ui.notify(f"Error processing {os.path.basename(file)}", type="negative", timeout=5000)
                continue
            finally:
                # Cleanup temp files
                if len(file_segments) > 1:
                    AudioSplitter.clear_temp_dir()
    
    except Exception as e:
        print(f"\nError during transcription: {str(e)}")
        ui.notify(f"Error: {str(e)}", type="negative", timeout=10000)
    
    finally:
        # Reset UI state
        viewmodel.is_transcribing = False
        viewmodel.button_abort_visible = False
        viewmodel.button_run_enabled = True
        viewmodel.spinner_progress_visibility = False
        
        # Update progress
        viewmodel.file_count = 0
        viewmodel.segment_count = 0
        viewmodel.segment_done_count = 0
        viewmodel.update_label_progress()
        
        # Show final status
        if viewmodel.abort_requested:
            ui.notify("Transcription aborted", type="warning", timeout=5000)
            print(f"\nTranscription aborted. Processed {len(processed_files)} of {len(files)} files.")
        elif processed_files:
            ui.notify("Transcription complete", type="positive", timeout=5000)
            print(f"\nTranscription complete. Successfully processed {len(processed_files)} files.")
        else:
            ui.notify("No files were processed", type="warning", timeout=5000)

@ui.page('/')
def main_page():
    global viewmodel
    
    # Initialize storage
    if 'selected_output_format' not in app.storage.general:
        app.storage.general['selected_output_format'] = ['xlsx', 'txt','vtt']
    if 'dark_mode' not in app.storage.general:
        app.storage.general['dark_mode'] = None
    if 'mute' not in app.storage.general:
        app.storage.general['mute'] = False
        
    dark_mode = ui.dark_mode().bind_value(app.storage.general, 'dark_mode')
    
    # Build UI
    with ui.column().classes('w-full'):
        with ui.row().classes('w-full items-center'):
            ui.icon('record_voice_over', color='primary').classes('text-4xl')
            ui.label('Swiss German Whisper Transcribe').classes('text-primary').style('font-size: 150%')
            ui.space()
            with ui.column():
                ui.button(icon='auto_mode', on_click=dark_mode.disable) \
                    .props('outline round').tooltip('automatic theme').bind_visibility_from(dark_mode, 'value', lambda mode: mode is None)
                ui.button(icon='light_mode', on_click=dark_mode.enable) \
                    .props('outline round').tooltip('light theme').bind_visibility_from(dark_mode, 'value', value=False)
                ui.button(icon='dark_mode', on_click=dark_mode.auto) \
                    .props('outline round').tooltip('dark theme').bind_visibility_from(dark_mode, 'value', value=True)
            with ui.column():
                ui.button(icon='volume_up', on_click=ViewModel.toggle_mute) \
                    .props('outline round').tooltip('play sound').bind_visibility_from(app.storage.general, 'mute', value=False)
                ui.button(icon='volume_off', on_click=ViewModel.toggle_mute) \
                    .props('outline round').tooltip('mute').bind_visibility_from(app.storage.general, 'mute', value=True)
        
        ui.button(icon='insert_drive_file', on_click=choose_files) \
            .bind_text_from(viewmodel, 'button_file_content').style('margin-top: 8px')
        
        ui.select(options=OUTPUT_FORMATS, label='output', multiple=True,
                 on_change=viewmodel.update_select_output_formats) \
            .classes('w-full') \
            .bind_value(app.storage.general, 'selected_output_format') \
            .props('use-chips')
        
        ui.label('Results are saved in the same directory as the original files.') \
            .style('color: #808080; font-style: italic; margin-top: 16px')
        
        with ui.row().classes('w-full justify-center gap-4'):
            start_button = ui.button(
                'start', 
                icon='auto_awesome',
                on_click=lambda: asyncio.create_task(transcribe_files(
                    viewmodel.selected_files,
                    app.storage.general['selected_output_format']
                ))
            ).bind_enabled_from(viewmodel, 'button_run_enabled')
            
            abort_button = ui.button(
                'abort', 
                icon='stop', 
                color='negative',
                on_click=abort_transcription
            ).bind_visibility_from(viewmodel, 'button_abort_visible')
        
        with ui.row().classes('w-full justify-center'):
            ui.spinner('dots', size='xl').bind_visibility_from(viewmodel, 'spinner_progress_visibility')
        
        ui.label().classes('w-full text-center') \
            .style('color: #808080; font-style: italic; white-space: pre-wrap') \
            .bind_text_from(viewmodel, 'label_progress_content')
        
        viewmodel.console_expansion = ui.expansion().classes('w-full')
        with viewmodel.console_expansion as expansion:
            ui.query('.nicegui-expansion .q-expansion-item__content').style('padding:0', replace='gap:0')
            with expansion.add_slot('header'):
                with ui.row().classes('w-full items-center'):
                    ui.label('console output').style('color: #808080')
                    ui.space()
            viewmodel.ui_log = ui.log(max_lines=100).classes("w-full h-40").style('white-space: pre-wrap')

# Suppress console output from NiceGUI
import logging
logging.getLogger('nicegui').setLevel(logging.WARNING)
logging.getLogger('watchfiles').setLevel(logging.WARNING)

# Initialize model with device selection
def initialize_model():
    # Check system capabilities
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"\nGPU VRAM available: {vram:.0f}MB")
        if vram < 4000:  # Less than 4GB VRAM
            print("Warning: Limited GPU memory. Using CPU instead for better reliability.")
            device = "cpu"
        else:
            device = "cuda"
    else:
        device = "cpu"
    
    return WhisperModelHandler(device=device)

# Combined startup handler
@app.on_startup
async def startup():
    global model, viewmodel
    
    # Start console reader and expand console
    asyncio.create_task(start_reading_console())
    await asyncio.sleep(0.5)  # Wait for UI to initialize
    
    if viewmodel.console_expansion:
        viewmodel.console_expansion.expand()
        print("Starting application...")
        
    # Verify ffmpeg installation
    from src.verify_ffmpeg import verify_ffmpeg
    if not verify_ffmpeg():
        viewmodel.button_run_enabled = False
        ui.notify("Please install ffmpeg as current user", type="negative", timeout=10000)
        return
    
    print(f"System memory: {psutil.Process().memory_info().rss / 1024**2:.1f}MB used")
    
    # Initialize model with proper device selection
    try:
        print("\nInitializing model...")
        model = initialize_model()
        
        print("\nLoading Swiss German specialized model...")
        print("This will download ~3GB of data on first run.")
        print("Subsequent runs will use the cached model.")
        model.load_model()
        
        if model.device == "cuda":
            print(f"Model loaded successfully on GPU. VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        else:
            print(f"Model loaded successfully on CPU. RAM usage: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")
            
        viewmodel.button_run_enabled = True
        
    except Exception as e:
        print(f"\nError during startup: {str(e)}")
        print("Application will continue running, but model will be loaded when needed.")
        viewmodel.button_run_enabled = False
        ui.notify("Model initialization failed. Will try again when transcribing.", type="warning", timeout=10000)

ui.run(
    title='Swiss German Whisper Transcribe',
    reload=False,
    native=True,
    window_size=[500,800],
    storage_secret='foobar',
    show=False,  # Suppress extra window
    dark=None,   # Use system theme
    port=random.randint(49152, 65535)  # Random port to avoid conflicts
)
