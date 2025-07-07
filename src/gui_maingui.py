# region Imports 

import os
import asyncio
import random
from typing import Optional, List
from queue import Queue
from nicegui import app, ui
from src.audio_config import (
    TIMESTAMPS_CONFIG,
    MODELS,
    APP_CONFIG,
    OUTPUT_FORMATS,
    WHISPER_LANGUAGES
)
import src.print_color as pc
from src.file_handler import choose_files
from src.gui_viewmodel import ViewModel

from src.logger_setup import gui_logger as logger

# endregion Imports


# region Utility Functions

def list_downloaded_models(cache_dir="model_cache"):
    """
    List downloaded models from the model_cache directory.
    Ignores hidden directories (starting with '.')
    Returns a list of available model names.
    """
    logger.debug(f"Listing downloaded models from {cache_dir}")
    base_path = os.path.join(os.path.dirname(__file__), "..", cache_dir)
    try:
        models = [
            d
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d)) and not d.startswith(".")
        ]
        logger.info(f"Found {len(models)} downloaded models")
        return models
    except FileNotFoundError:
        logger.warning(f"Model cache directory not found: {base_path}")
        return []


def mark_downloaded_models(model_name, downloaded_models):
    """Add a marker to downloaded models."""
    cache_name = f"models--{model_name.replace('/', '--')}"
    return f"{'✓' if cache_name in downloaded_models else 'X'} {model_name}"

# endregion Utility Functions


#region MainGUI

class MainGUI:
    def __init__(self):
        self.app_name = APP_CONFIG["name"]
        self.viewmodel = ViewModel()
        self.model = None
        self.startup_printed = False
        pc.init()
        self.setup_storage()
        
        # Initialize UI elements
        self.button_file_content = "choose audio / video files"
        self.label_progress_content = ""
        self.button_run_enabled = False
        self.button_abort_visible = False
        self.spinner_progress_visibility = False
        
        # UI update queue
        self.ui_update_queue = Queue()
        self._setup_ui_timer()

    def update_ui(self, message, notify_type: Optional[str] = None) -> None:
        """Queue a UI update to be processed in the main thread."""
        if not isinstance(message, str):
            logger.warning(f"Invalid message type: {type(message)}")
            message = str(message)

        logger.debug(f"UI Update: {message} (type: {notify_type})")

        if notify_type:
            self.ui_update_queue.put({
                "type": "notify",
                "message": message,
                "notify_type": notify_type
            })
        self.ui_update_queue.put({
            "type": "log",
            "message": message
        })

    def toggle_mute(self):
        logger.debug("Toggling mute state")
        app.storage.general["mute"] = not app.storage.general["mute"]
        self.update_ui("Sound " + ("unmuted" if not app.storage.general["mute"] else "muted"), "info")

    def toggle_dark_mode(self):
        logger.debug("Toggling dark mode")
        app.storage.general["dark"] = not app.storage.general.get("dark", False)
        ui.dark_mode(app.storage.general["dark"])
        self.update_ui("Dark mode " + ("enabled" if app.storage.general["dark"] else "disabled"), "info")

    async def play_sound_effect(self, sound_file: str):
        """Generic sound effect player"""
        if app.storage.general.get("mute", False):
            return

        try:
            if not os.path.isfile(sound_file):
                sound_file = os.path.join("_internal", sound_file)
            if os.path.isfile(sound_file):
                proc = await asyncio.create_subprocess_exec(
                    "ffplay", "-nodisp", "-autoexit", "-hide_banner",
                    "-loglevel", "quiet", sound_file,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await proc.wait()
        except Exception as e:
            logger.warning(f"Sound effect error: {e}")

    def _setup_ui_timer(self):
        """Setup UI update timer"""
        ui.timer(0.1, self.process_ui_updates)

    async def process_ui_updates(self):
        """Process UI updates asynchronously"""
        while True:
            try:
                update = await self.ui_update_queue.get()
                if update["type"] == "notify":
                    ui.notify(update["message"], type=update["notify_type"])
                elif update["type"] == "log":
                    if hasattr(self, 'ui_log'):
                        self.ui_log.push(update["message"])
            except Exception as e:
                logger.error(f"Error processing UI update: {e}")

    def update_ui(self, message, notify_type: Optional[str] = None) -> None:
        """Queue a UI update to be processed in the main thread."""
        if not isinstance(message, str):
            logger.warning(f"Invalid message type: {type(message)}")
            message = str(message)

        logger.debug(f"UI Update: {message} (type: {notify_type})")

        if notify_type:
            self.ui_update_queue.put({
                "type": "notify",
                "message": message,
                "notify_type": notify_type
            })
        self.ui_update_queue.put({
            "type": "log",
            "message": message
        })

    def update_button_states(self):
        """Update button states based on file and format selection"""
        logger.debug("Updating button states")
        if self.viewmodel.is_transcribing:
            self.button_run_enabled = False
            self.button_abort_visible = True
        else:
            self.button_abort_visible = False
            if self.viewmodel.selected_files is None or len(self.viewmodel.selected_files) == 0:
                self.button_file_content = "choose audio / video files"
                self.button_run_enabled = False
            else:
                self.button_file_content = (
                    "1 File selected"
                    if len(self.viewmodel.selected_files) == 1
                    else f"{len(self.viewmodel.selected_files)} files selected"
                )
                ui.label(
                    f"Files: \n{self.viewmodel.selected_files}"
                ).style("color: #808080; font-style: italic; margin-top: 16px")
                formats = app.storage.general.get("selected_output_format", [])
                self.button_run_enabled = bool(formats) and bool(self.viewmodel.selected_files)

    def update_progress_ui(self):
        """Update progress-related UI elements"""
        progress = self.viewmodel.get_progress()
        self.label_progress_content = f"Progress: {progress['segment_done_count'] / progress['segment_count'] * 100:.1f}%"
        self.spinner_progress_visibility = progress['segment_count'] > 0

    def setup_storage(self):
        """Initialize default storage values"""
        if "selected_output_format" not in app.storage.general:
            app.storage.general["selected_output_format"] = ["txt"]
            self.selected_output_formats = app.storage.general["selected_output_format"]
            self.update_button_states()

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

    def build_main_page(self):
        """Build the main page UI"""
        logger.info("Main page initialization started")

        with ui.column().classes("w-full"):
            self.build_header()
            self.build_file_selector()
            self.build_timestamp_controls()
            self.build_format_selector()
            self.build_model_selector()
            self.build_language_selector()
            self.build_info_label()
            self.build_control_buttons()
            self.build_progress_indicators()

        logger.info("Main page initialization completed")

    def build_header(self):
        """Build the header section"""
        with ui.row().classes("w-full items-center"):
            ui.icon("record_voice_over", color="primary").classes("text-4xl")
            ui.label(self.app_name).classes("text-primary").style("font-size: 150%")
            ui.space()
            with ui.row():
                self.build_sound_buttons()
                self.build_theme_buttons()

    def build_sound_buttons(self):
        """Build sound control buttons"""
        ui.button(icon="volume_up", on_click=self.toggle_mute)\
            .props("outline round")\
            .tooltip("play sound")\
            .bind_visibility_from(app.storage.general, "mute", value=False)
        ui.button(icon="volume_off", on_click=self.toggle_mute)\
            .props("outline round")\
            .tooltip("mute")\
            .bind_visibility_from(app.storage.general, "mute", value=True)

    def build_theme_buttons(self):
        """Build theme control buttons"""
        ui.button(icon="light_mode", on_click=self.toggle_dark_mode)\
            .props("outline round")\
            .tooltip("light mode")\
            .bind_visibility_from(app.storage.general, "dark", value=True)
        ui.button(icon="dark_mode", on_click=self.toggle_dark_mode)\
            .props("outline round")\
            .tooltip("dark mode")\
            .bind_visibility_from(app.storage.general, "dark", value=False)

    def build_file_selector(self):
        """Build file selector button"""
        ui.button(
            icon="insert_drive_file",
            on_click=lambda: choose_files(self.viewmodel)
        ).bind_text_from(self.viewmodel, "button_file_content").style("margin-top: 8px")

    def build_timestamp_controls(self):
        """Build timestamp control checkboxes"""
        ui.checkbox(
            "Include regular timestamps",
            on_change=lambda e: self.update_timestamp_settings("use_timestamps", e.value)
        ).bind_value_from(TIMESTAMPS_CONFIG, "use_timestamps")

        ui.checkbox(
            "Enhance timestamps with WhisperX",
            on_change=lambda e: self.update_timestamp_settings("use_whisperx", e.value)
        ).bind_value_from(TIMESTAMPS_CONFIG, "use_whisperx")

    def build_format_selector(self):
        """Build output format selector"""
        ui.select(
            options=OUTPUT_FORMATS,
            label="output",
            multiple=True,
            on_change=lambda e: self.update_output_formats(e.value),
        ).classes("w-full")\
         .bind_value(app.storage.general, "selected_output_format")\
         .props("use-chips")

    def build_model_selector(self):
        """Build model selector dropdown"""
        downloaded_models = list_downloaded_models()
        model_options = [mark_downloaded_models(m, downloaded_models) for m in MODELS]
        stored_model = app.storage.general.get("selected_model")
        initial_model = stored_model if stored_model in MODELS else MODELS[0]
        model_mapping = {
            mark_downloaded_models(m, downloaded_models): m for m in MODELS
        }
        ui.select(
            options=model_options,
            label="model",
            value=mark_downloaded_models(initial_model, downloaded_models),
            on_change=lambda e: self.update_model_selection(model_mapping[e.value]),
        ).classes("w-full")

    def build_language_selector(self):
        """Build language selector dropdown"""
        ui.select(
            options=WHISPER_LANGUAGES,
            label="language",
            on_change=lambda e: self.update_language_selection(e.value),
        ).classes("w-full").bind_value(app.storage.general, "selected_language")

    def update_output_formats(self, formats: List[str]):
        """Update output formats"""
        app.storage.general["selected_output_format"] = formats
        self.viewmodel.update_output_formats(formats)

    def update_model_selection(self, model: str):
        """Update selected model"""
        app.storage.general["selected_model"] = model
        self.viewmodel.update_model_selection(model)

    def update_language_selection(self, language: str):
        """Update selected language"""
        app.storage.general["selected_language"] = language
        self.viewmodel.update_language_selection(language)

    def build_info_label(self):
        """Build information label"""
        ui.label(
            "Results are saved in the same directory as the original files."
        ).style("color: #808080; font-style: italic; margin-top: 16px")

    def build_control_buttons(self):
        """Build control buttons (start/abort)"""
        with ui.row().classes("w-full justify-center gap-4"):
            ui.button(
                "start",
                icon="auto_awesome",
                on_click=lambda: asyncio.create_task(self.start_transcription()),
            ).props("no-caps").bind_enabled_from(self, "button_run_enabled")

            ui.button(
                "abort",
                icon="stop",
                color="negative",
                on_click=self.abort_transcription,
            ).bind_visibility_from(self, "button_abort_visible")

    def build_progress_indicators(self):
        """Build progress indicators"""
        with ui.row().classes("w-full justify-center"):
            ui.spinner("dots", size="xl").bind_visibility_from(
                self, "spinner_progress_visibility"
            )
            ui.label().classes("w-full text-center").style(
                "color: #808080; font-style: italic; white-space: pre-wrap"
            ).bind_text_from(self, "label_progress_content")

    async def start_transcription(self):
        """Start transcription process"""
        self.update_ui("Starting transcription...", "info")
        result = await self.viewmodel.start_transcription()
        if result:
            self.update_ui("Transcription completed successfully", "positive")
        else:
            self.update_ui("Transcription failed or was aborted", "negative")
        self.update_button_states()
        self.update_progress_ui()

    def abort_transcription(self):
        """Abort ongoing transcription"""
        self.viewmodel.abort_transcription()
        self.update_ui("Aborting transcription...", "warning")

    def update_timestamp_settings(self, setting: str, value: bool):
        """Update timestamp settings ensuring mutual exclusion"""
        if value:
            if setting == "use_timestamps":
                TIMESTAMPS_CONFIG.update({
                    "use_timestamps": True,
                    "use_whisperx": False
                })
            else:  # use_whisperx
                TIMESTAMPS_CONFIG.update({
                    "use_timestamps": False,
                    "use_whisperx": True
                })
        else:
            TIMESTAMPS_CONFIG[setting] = False

    def run(self):
        """Run the application"""
        print("Debug: Starting NiceGUI server")
        try:
            self.build_main_page()

            # Start the NiceGUI server using asyncio
            ui.run(
                title=self.app_name,
                reload=False,
                native=True,
                window_size=[600, 800],
                storage_secret=self.generate_secret(),
                show=True,
                dark=False,
                port=random.randint(49152, 65535),
                show_welcome_message=False,
            )
            print("Debug: NiceGUI server started successfully")
        except Exception as e:
            print(f"Debug: Error starting NiceGUI server - {str(e)}")

    @staticmethod
    def generate_secret(length=16):
        """Generate a secret key for storage"""
        import string
        import secrets
        characters = string.ascii_letters + string.digits + string.punctuation
        return "".join(secrets.choice(characters) for _ in range(length))
    
#endregion MainGUI
