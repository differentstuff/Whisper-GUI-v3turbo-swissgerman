import os
import sys
from nicegui import app, ui, events
import asyncio
from io import StringIO
from logging import getLogger, StreamHandler
from pydub import AudioSegment
import math
import glob
import openpyxl
import simpleaudio
from typing import List, Optional

class ViewModel:
    def __init__(self):
        self.button_file_content = 'choose audio files'
        self.selected_files = None
        self.label_progress_content = ''
        self.button_run_enabled = False
        self.button_abort_visible = False
        self.spinner_progress_visibility = False
        self.ui_log = ui.log()
        self.console_expansion = None
        self.file_count = 0
        self.file_count_old = 0
        self.segment_count = 0
        self.segment_done_count = 0
        self.segment_current_progress = 0
        self.selected_output_formats = []
        self.is_transcribing = False
        self.abort_requested = False
            
    def update_label_progress(self):
        if self.file_count <= 0:
            self.spinner_progress_visibility = False
            self.label_progress_content = ''
            self.segment_current_progress = 0
            self.segment_count = 0
            self.segment_done_count = 0
            ui.notify('finished transcribing')
            self.play_sound_effect_finished()
        else:
            if self.file_count_old == 0:
                self.spinner_progress_visibility = True
            info = ''
            if self.file_count == 1:
                info += f'transcribing {self.file_count} file'
            else:
                info += f'transcribing {self.file_count} files'
            if self.segment_count == 1:
                info += f' ({self.segment_done_count} / {self.segment_count} segment done)'
            else:
                info += f' ({self.segment_done_count} / {self.segment_count} segments done)'
            if self.segment_current_progress > 0:
                info += f'\ncurrent segment {self.segment_current_progress}% done'
            self.label_progress_content = info
        self.file_count_old = self.file_count
    
    def update_buttons(self):
        if self.selected_files is None or len(self.selected_files) == 0:
            self.button_run_enabled = False
            self.button_file_content = 'choose audio / video files'
        else:
            if len(self.selected_files) == 1:
                self.button_file_content = '1 file selected'
            else:    
                self.button_file_content = f'{len(self.selected_files)} files selected'
            if self.selected_output_formats is None or len(self.selected_output_formats) == 0:
                self.button_run_enabled = False
            else:
                self.button_run_enabled = True

    def update_select_output_formats(self, e: events.ValueChangeEventArguments):
        self.selected_output_formats = e.value
        self.update_buttons()

    def toggle_mute():
        app.storage.general['mute'] = not app.storage.general['mute']

    def play_sound_effect_finished(self):
        if not app.storage.general.get('mute', False):
            sound_effect_path = "sound_effect_finished.wav"
            if not os.path.isfile(sound_effect_path):
                sound_effect_path = os.path.join("_internal", "sound_effect_finished.wav")
            if sound_effect_path is not None:
                wave_obj = simpleaudio.WaveObject.from_wave_file(sound_effect_path)
                wave_obj.play()

class AudioSplitter:
    @staticmethod
    def split_audio(file: str, max_size: int = 25000000, overlap_ms: int = 2000) -> List[tuple]:
        split_files = []
        segment_count = AudioSplitter.get_segment_count(file, max_size)
        if segment_count > 1:
            temp_dir = AudioSplitter.create_temp_dir()
            song = AudioSegment.from_file(file)
            print(f'\nsplitting {file} into {segment_count} parts')
            segment_length_ms = len(song) / segment_count
            overlap = overlap_ms if segment_length_ms >= overlap_ms else 0
            
            for i in range(segment_count):
                seg_start_ms = math.floor(i * segment_length_ms - overlap / 2) if i > 0 else 0
                seg_end_ms = math.ceil((i + 1) * segment_length_ms + overlap / 2) if i < segment_count - 1 else None
                
                segment = song[seg_start_ms:seg_end_ms]
                segment_filename = os.path.join(
                    temp_dir, 
                    f"{os.path.splitext(os.path.basename(file))[0]}_segment{i}{os.path.splitext(file)[1]}"
                )
                segment.export(segment_filename)
                print(f'\n  > saved segment {segment_filename}')
                split_files.append((segment_filename, seg_start_ms))
            print(f'\nfinished splitting {file}')
        else:
            split_files.append((file, 0))
        return split_files
        
    @staticmethod
    def get_segment_count(file: str, max_size: int = 25000000) -> int:
        file_size = os.path.getsize(file)
        return math.ceil(file_size / max_size) if file_size > max_size else 1

    @staticmethod
    def get_temp_dir() -> str:
        if getattr(sys, 'frozen', False):
            return os.path.join(os.path.dirname(sys.executable), '.temp')
        elif __file__:
            return os.path.join(os.path.dirname(__file__), '.temp')
        return os.path.join(os.getcwd(), '.temp')

    @staticmethod
    def create_temp_dir() -> str:
        temp_dir = AudioSplitter.get_temp_dir()
        print(f'\ntemp dir: {temp_dir}')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        return temp_dir

    @staticmethod
    def clear_temp_dir():
        temp_dir = AudioSplitter.get_temp_dir()
        files = glob.glob(os.path.join(temp_dir,'*'))
        for f in files:
            os.remove(f)

class TranscriptionHandler:
    @staticmethod
    def save_result(result: dict, output_formats: List[str], file_path: str):
        output_dir = os.path.dirname(os.path.realpath(file_path))
        
        for ext in output_formats:
            output_filename = f"{os.path.splitext(file_path)[0]}.{ext}"
            
            if ext == 'xlsx':
                wb = openpyxl.Workbook()
                sheet = wb.active
                sheet.cell(row=1, column=1).value = 'start'
                sheet.cell(row=1, column=2).value = 'end'
                sheet.cell(row=1, column=3).value = 'text'
                
                for i, segment in enumerate(result["segments"], 2):
                    sheet.cell(row=i, column=1).value = segment["start"]
                    sheet.cell(row=i, column=2).value = segment["end"]
                    sheet.cell(row=i, column=3).value = segment["text"]
                
                wb.save(os.path.join(output_dir, output_filename))
            else:
                # Handle other formats (txt, srt, vtt, etc.)
                with open(os.path.join(output_dir, output_filename), 'w', encoding='utf-8') as f:
                    if ext == 'txt':
                        f.write(result['text'])
                    elif ext in ['srt', 'vtt']:
                        for i, segment in enumerate(result['segments'], 1):
                            f.write(f"{i}\n")
                            start = TranscriptionHandler.format_timestamp(segment['start'])
                            end = TranscriptionHandler.format_timestamp(segment['end'])
                            f.write(f"{start} --> {end}\n")
                            f.write(f"{segment['text']}\n\n")

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

async def start_reading_console():
    string_io = StringIO()
    sys.stdout = string_io
    sys.stderr = string_io
    stream_handler = StreamHandler(string_io)
    stream_handler.setLevel("DEBUG")
    logger = getLogger(__name__)
    logger.setLevel("DEBUG")
    logger.addHandler(stream_handler)
    
    while True:
        await asyncio.sleep(1)
        ui.log().push(string_io.getvalue())
        string_io.truncate(0)
        string_io.seek(0)
