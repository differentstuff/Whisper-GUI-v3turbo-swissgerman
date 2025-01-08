import asyncio
from nicegui import app, ui
from src.model_handler import WhisperModelHandler
from src.gui_handler import ViewModel, AudioSplitter, TranscriptionHandler, start_reading_console

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
    global viewmodel, model
    
    # Reset abort flag and update UI state
    viewmodel.abort_requested = False
    viewmodel.is_transcribing = True
    viewmodel.button_abort_visible = True
    viewmodel.button_run_enabled = False
    
    # Auto-expand console
    viewmodel.console_expansion.expand()
    
    viewmodel.file_count += len(files)
    viewmodel.selected_files = None
    viewmodel.update_buttons()
    
    # Calculate total segments
    total_segments = sum(AudioSplitter.get_segment_count(file) for file in files)
    viewmodel.segment_count += total_segments
    viewmodel.update_label_progress()
    
    try:
        # Load model if not loaded
        if model.processor is None:
            print("\nLoading Swiss German specialized model, this might take a while...")
            model.load_model()
        
        # Process each file
        for file in files:
            if viewmodel.abort_requested:
                print("\nTranscription aborted by user.")
                break
            
            file_segments = AudioSplitter.split_audio(file)
            results = []
            
            for segment_file, start_time in file_segments:
                result = model.transcribe(segment_file)
                
                # Adjust timestamps for segments
                if start_time > 0:
                    for segment in result["segments"]:
                        segment["start"] += start_time / 1000  # ms to s
                        segment["end"] += start_time / 1000
                
                results.append(result)
                viewmodel.segment_done_count += 1
                viewmodel.update_label_progress()
            
            # Combine results if file was split
            if len(results) > 1:
                combined_result = results[0].copy()
                combined_result["text"] = "\n".join(r["text"] for r in results)
                combined_result["segments"] = []
                for r in results:
                    combined_result["segments"].extend(r["segments"])
                TranscriptionHandler.save_result(combined_result, output_format, file)
            else:
                TranscriptionHandler.save_result(results[0], output_format, file)
            
            # Cleanup temp files
            if len(file_segments) > 1:
                AudioSplitter.clear_temp_dir()
    
    except Exception as e:
        print(f"\nError during transcription: {str(e)}")
        ui.notify(f"Error: {str(e)}", type="negative")
    
    finally:
        # Reset UI state
        viewmodel.is_transcribing = False
        viewmodel.button_abort_visible = False
        viewmodel.button_run_enabled = True
        
        # Update progress
        viewmodel.file_count -= len(files)
        viewmodel.segment_count -= total_segments
        viewmodel.update_label_progress()

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
            ui.button('start', icon='auto_awesome',
                     on_click=lambda: transcribe_files(viewmodel.selected_files,
                                                     app.storage.general['selected_output_format'])) \
                .bind_enabled_from(viewmodel, 'button_run_enabled')
            
            ui.button('abort', icon='stop', color='negative',
                     on_click=lambda: abort_transcription()) \
                .bind_visibility_from(viewmodel, 'button_abort_visible')
        
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

app.on_startup(start_reading_console)
ui.run(title='Swiss German Whisper Transcribe', reload=False, native=True, window_size=[500,800], storage_secret='foobar')
