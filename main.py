import random
import os
import torch
import psutil
import asyncio
from nicegui import app, ui
from src.model_handler import WhisperModelHandler
from src.gui_handler import ViewModel, start_reading_console
from src.file_handler import (
    OUTPUT_FORMATS,
    choose_files
)

# Suppress console output from NiceGUI
import logging
logging.getLogger('nicegui').setLevel(logging.WARNING)
logging.getLogger('watchfiles').setLevel(logging.WARNING)

# Suppress unnecessary output
os.environ['PYTHONWARNINGS'] = 'ignore'

viewmodel = ViewModel()
model = None  # Will be initialized in startup

@ui.page('/')
def main_page():
    global viewmodel
    
    # Initialize storage with defaults
    if 'selected_output_format' not in app.storage.general:
        app.storage.general['selected_output_format'] = ['txt']  # Start with simple text output
    viewmodel.selected_output_formats = app.storage.general['selected_output_format']
    viewmodel.update_button_states()  # Update button state based on initial format
    
    if 'mute' not in app.storage.general:
        app.storage.general['mute'] = False
    
    # Build UI
    with ui.column().classes('w-full'):
        with ui.row().classes('w-full items-center'):
            ui.icon('record_voice_over', color='primary').classes('text-4xl')
            ui.label('Swiss German Whisper Transcribe').classes('text-primary').style('font-size: 150%')
            ui.space()
            with ui.column():
                ui.button(icon='volume_up', on_click=ViewModel.toggle_mute) \
                    .props('outline round').tooltip('play sound').bind_visibility_from(app.storage.general, 'mute', value=False)
                ui.button(icon='volume_off', on_click=ViewModel.toggle_mute) \
                    .props('outline round').tooltip('mute').bind_visibility_from(app.storage.general, 'mute', value=True)
        
        ui.button(icon='insert_drive_file', on_click=lambda: choose_files(viewmodel)) \
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
                on_click=lambda: asyncio.create_task(viewmodel.start_transcription())
            ).props('no-caps').bind_enabled_from(viewmodel, 'button_run_enabled')
            
            abort_button = ui.button(
                'abort', 
                icon='stop', 
                color='negative',
                on_click=viewmodel.abort_transcription
            ).bind_visibility_from(viewmodel, 'button_abort_visible')
        
        with ui.row().classes('w-full justify-center'):
            ui.spinner('dots', size='xl').bind_visibility_from(viewmodel, 'spinner_progress_visibility')
        
        ui.label().classes('w-full text-center') \
            .style('color: #808080; font-style: italic; white-space: pre-wrap') \
            .bind_text_from(viewmodel, 'label_progress_content')
        
        # Console output section
        viewmodel.console_expansion = ui.expansion('Console Output', value=True).classes('w-full')
        with viewmodel.console_expansion as expansion:
            ui.query('.nicegui-expansion .q-expansion-item__content').style('padding:0', replace='gap:0')
            viewmodel.ui_log = ui.log(max_lines=1000).classes("w-full h-60").style('white-space: pre-wrap')

# Initialize model with device selection
def initialize_model(device: str):
    """Initialize the model with the specified device"""
    try:
        print(f"Initializing model handler on {device.upper()}...")
        return WhisperModelHandler(device=device)
    except Exception as e:
        print(f"\n❌ Error initializing model: {str(e)}")
        ui.notify("Error initializing model", type="negative", timeout=10000)
        return None

# Function to process console messages
async def process_console_messages():
    while True:
        viewmodel.process_console_messages()
        await asyncio.sleep(0.1)

# Combined startup handler
@app.on_startup
async def startup():
    global viewmodel
    
    print("\nStarting application initialization...")
    
    # Initialize UI and console
    if viewmodel.console_expansion:
        viewmodel.console_expansion.expand()
        await asyncio.sleep(0.1)  # Wait for expansion animation
    
    # Start console reader
    console_task = asyncio.create_task(start_reading_console(viewmodel))
    
    # Start console message processor
    asyncio.create_task(process_console_messages())
    
    await asyncio.sleep(0.5)  # Give console time to initialize
    
    # Clear any previous output
    if viewmodel.ui_log:
        viewmodel.ui_log.clear()
    
    print("\n=== Swiss German Whisper Transcribe ===")
    print("Initializing system...")
    
    # Determine device based on system capabilities
    device = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print("\n=== System Information ===")
        print(f"GPU: {gpu_name}")
        print(f"VRAM Available: {vram:.0f}MB")
        print(f"RAM Usage: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")
        
        if vram >= 4000:  # Need at least 4GB VRAM
            device = "cuda"
            print("✓ Using GPU for processing")
        else:
            print("⚠️  Limited GPU memory - using CPU for better reliability")
    else:
        print("\n=== System Information ===")
        print("• CPU Only Mode")
        print(f"RAM Usage: {psutil.Process().memory_info().rss / 1024**2:.1f}MB")
    
    # Verify ffmpeg installation
    from src.verify_ffmpeg import verify_ffmpeg
    if not verify_ffmpeg():
        viewmodel.button_run_enabled = False
        ui.notify("Please install ffmpeg as current user", type="negative", timeout=10000)
        return
    
    # Initialize model handler
    try:
        print("\n=== Model Initialization ===")
        viewmodel.model = initialize_model(device)
        if viewmodel.model is None:
            print("❌ Model initialization failed")
            viewmodel.button_run_enabled = False
            return
            
        # Don't load model at startup - wait for first transcription
        viewmodel.button_run_enabled = True
        print("✓ Model handler initialized - will load model when needed")
            
    except Exception as e:
        print(f"\nError during startup: {str(e)}")
        viewmodel.button_run_enabled = False
        ui.notify("Startup failed. Please check the console for details.", type="negative", timeout=10000)

ui.run(
    title='Swiss German Whisper Transcribe',
    reload=False,
    native=True,
    window_size=[500,800],
    storage_secret='foobar',
    show=False,  # Suppress extra window
    dark=True,   # Use dark theme
    port=random.randint(49152, 65535)  # Random port to avoid conflicts
)
