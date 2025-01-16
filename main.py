# region Imports
import random
import os
import string
import torch
import psutil
import asyncio
import secrets
from nicegui import app, ui
from src.model_handler import WhisperModelHandler
from src.gui_handler import (
    ViewModel,
    list_downloaded_models,
    mark_downloaded_models,
    LANGUAGES,
)
from src.file_handler import OUTPUT_FORMATS, choose_files
import src.print_color as pc

import logging
from src.logger_setup import main_logger as logger

# endregion Imports


# region Constants
MODELS = [
    "nizarmichaud/whisper-large-v3-turbo-swissgerman",
    "openai/whisper-large-v3-turbo",
    "aiola/whisper-medusa-multilingual",
]
# endregion Constants


# region Initialization
logging.getLogger("nicegui").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
app_name = "Whisper Transcribe v2"
startup_printed = False
pc.init()  # Initialize colorama
# Suppress unnecessary output
os.environ["PYTHONWARNINGS"] = "ignore"
viewmodel = ViewModel()
model = None  # Model will be initialized in startup
# endregion Initialization


# region Functions
@ui.page("/")
def main_page():
    logger.info("Main page initialization started")
    global viewmodel

    # Initialize storage with defaults
    if "selected_output_format" not in app.storage.general:
        logger.info("Initializing default output format")
        app.storage.general["selected_output_format"] = [
            "txt"
        ]  # Start with simple text output
    viewmodel.selected_output_formats = app.storage.general["selected_output_format"]
    viewmodel.update_button_states()  # Update button state based on initial format

    if "selected_language" not in app.storage.general:
        app.storage.general["selected_language"] = "Auto"

    if "selected_model" not in app.storage.general:
        app.storage.general["selected_model"] = MODELS[0]

    if "mute" not in app.storage.general:
        app.storage.general["mute"] = False

    if "dark" not in app.storage.general:
        app.storage.general["dark"] = False

    if "timestamps_enabled" not in app.storage.general:
        app.storage.general["timestamps_enabled"] = False

    # Build UI
    with ui.column().classes("w-full"):
        with ui.row().classes("w-full items-center"):
            ui.icon("record_voice_over", color="primary").classes("text-4xl")
            ui.label(app_name).classes("text-primary").style("font-size: 150%")
            ui.space()
            with ui.row():
                ui.button(icon="volume_up", on_click=viewmodel.toggle_mute).props(
                    "outline round"
                ).tooltip("play sound").bind_visibility_from(
                    app.storage.general, "mute", value=False
                )
                ui.button(icon="volume_off", on_click=viewmodel.toggle_mute).props(
                    "outline round"
                ).tooltip("mute").bind_visibility_from(
                    app.storage.general, "mute", value=True
                )
                ui.button(icon="light_mode", on_click=viewmodel.toggle_dark_mode).props(
                    "outline round"
                ).tooltip("light mode").bind_visibility_from(
                    app.storage.general, "dark", value=True
                )
                ui.button(icon="dark_mode", on_click=viewmodel.toggle_dark_mode).props(
                    "outline round"
                ).tooltip("dark mode").bind_visibility_from(
                    app.storage.general, "dark", value=False
                )

        ui.button(
            icon="insert_drive_file", on_click=lambda: choose_files(viewmodel)
        ).bind_text_from(viewmodel, "button_file_content").style("margin-top: 8px")

        # Timestamps checkbox
        ui.checkbox(
            "Include timestamps", on_change=viewmodel.toggle_timestamps
        ).bind_value(app.storage.general, "timestamps_enabled").classes("w-full")

        ui.select(
            options=OUTPUT_FORMATS,
            label="output",
            multiple=True,
            on_change=viewmodel.update_select_output_formats,
        ).classes("w-full").bind_value(
            app.storage.general, "selected_output_format"
        ).props(
            "use-chips"
        )

        # Model selection dropdown
        downloaded_models = list_downloaded_models()
        model_options = [mark_downloaded_models(m, downloaded_models) for m in MODELS]
        stored_model = app.storage.general.get("selected_model")
        initial_model = stored_model if stored_model in MODELS else MODELS[0]
        # Create a mapping of marked options to actual model names for the select component
        model_mapping = {
            mark_downloaded_models(m, downloaded_models): m for m in MODELS
        }
        ui.select(
            options=model_options,
            label="model",
            value=mark_downloaded_models(initial_model, downloaded_models),
            on_change=lambda e: app.storage.general.__setitem__(
                "selected_model", model_mapping[e.value]
            ),
        ).classes("w-full")

        # Language selection dropdown
        ui.select(
            options=LANGUAGES,
            label="language",
            on_change=viewmodel.update_language_selection,
        ).classes("w-full").bind_value(app.storage.general, "selected_language")

        ui.label(
            "Results are saved in the same directory as the original files."
        ).style("color: #808080; font-style: italic; margin-top: 16px")

        with ui.row().classes("w-full justify-center gap-4"):
            start_button = (
                ui.button(
                    "start",
                    icon="auto_awesome",
                    on_click=lambda: asyncio.create_task(
                        viewmodel.start_transcription()
                    ),
                )
                .props("no-caps")
                .bind_enabled_from(viewmodel, "button_run_enabled")
            )

            abort_button = ui.button(
                "abort",
                icon="stop",
                color="negative",
                on_click=viewmodel.abort_transcription,
            ).bind_visibility_from(viewmodel, "button_abort_visible")

        with ui.row().classes("w-full justify-center"):
            ui.spinner("dots", size="xl").bind_visibility_from(
                viewmodel, "spinner_progress_visibility"
            )

        ui.label().classes("w-full text-center").style(
            "color: #808080; font-style: italic; white-space: pre-wrap"
        ).bind_text_from(viewmodel, "label_progress_content")

    logger.info("Main page initialization completed")


# Initialize model with device selection
def initialize_model(device: str):
    """Initialize the model with the specified device"""
    try:
        logger.info(f"Initializing model handler on {device}")
        pc.print_success(f"Initializing model handler on {device.upper()}")
        model = WhisperModelHandler(device=device)
        logger.info("Model handler initialized successfully")
        return model
    except Exception as e:
        error_msg = f"Error initializing model: {str(e)}"
        logger.error(error_msg, exc_info=True)
        pc.print_error(f"\nX {error_msg}")
        ui.notify("Error initializing model", type="negative", timeout=10000)
        return None


# Combined startup handler
@app.on_startup
async def startup():
    global viewmodel, startup_printed
    logger.info("Application startup initiated")

    # Only print startup message once
    if not startup_printed:
        logger.info("Application initialization started")
        pc.print_header("Whisper Transcribe v2")
        pc.print_info("Starting application initialization... Please wait")
        startup_printed = True

    # Clear any previous output
    if viewmodel.ui_log:
        viewmodel.ui_log.clear()

    # Determine device based on system capabilities
    pc.print_header("System Information")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        logger.info(f"GPU detected: {gpu_name} with {vram:.0f}MB VRAM")
        if vram >= 3000:  # Need at least 3GB VRAM
            device = "cuda"
            logger.info("Using GPU for processing")
            pc.print_success("Using GPU for processing")
        else:
            device = "cpu"
            pc.print_warning("Limited GPU memory - using CPU for better reliability")
            logger.warning("Limited GPU memory - falling back to CPU")
        pc.print_info(f"GPU: {gpu_name}")
        pc.print_info(f"VRAM Available: {vram:.0f}MB")
        pc.print_info(
            f"RAM Usage: {psutil.Process().memory_info().rss / 1024**2:.1f}MB"
        )
    else:
        device = "cpu"
        logger.info("No GPU detected - using CPU mode")
        pc.print_info("• CPU Only Mode")
        pc.print_info(
            f"RAM Usage: {psutil.Process().memory_info().rss / 1024**2:.1f}MB"
        )

    # Verify ffmpeg installation
    from src.verify_ffmpeg import verify_ffmpeg

    if not verify_ffmpeg():
        logger.error("FFmpeg verification failed")
        viewmodel.button_run_enabled = False
        ui.notify("\nPlease install ffmpeg", type="negative", timeout=10000)
        return

    # Initialize model handler
    try:
        pc.print_header("Model Initialization")
        viewmodel.model = initialize_model(device)
        if viewmodel.model is None:
            pc.print_warning("Model initialization failed")
            viewmodel.button_run_enabled = False
            return

        # Don't load model at startup - wait for first transcription
        viewmodel.button_run_enabled = True
        pc.print_success("Model handler initialized - will load model when needed\n")

    except Exception as e:
        pc.print_error(f"\nError during startup: {str(e)}")
        viewmodel.button_run_enabled = False
        ui.notify(
            "Startup failed. Please check the console for details.",
            type="negative",
            timeout=10000,
        )


def generate_secret(length=16):
    characters = string.ascii_letters + string.digits + string.punctuation
    return "".join(secrets.choice(characters) for _ in range(length))


# endregion Functions


# region Runtime
ui.run(
    title=app_name,
    reload=False,
    native=True,
    window_size=[500, 700],
    storage_secret=generate_secret(),
    show=False,  # Suppress extra window
    dark=False,  # Use light theme
    port=random.randint(49152, 65535),  # Random port to avoid conflicts
    show_welcome_message=False,  # suppress NiceGUI's startup message
)
# endregion Runtime
