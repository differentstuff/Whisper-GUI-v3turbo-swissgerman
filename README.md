# Swiss German Whisper Transcribe

A GUI application for transcribing Swiss German audio/video files using the specialized Whisper Large V3 Turbo model.

## Features

- User-friendly GUI interface
- Supports multiple audio and video formats
- Automatic file splitting for large files
- Multiple output formats (XLSX, SRT, TXT, VTT, TSV, JSON)
- Dark/Light mode support
- Progress tracking with sound notification
- 100% local processing (no internet needed after model download)

## Installation & Usage

### Windows

1. Install Python 3.8 or newer from [python.org](https://www.python.org/downloads/)
2. Install ffmpeg by running in Command Prompt (as Administrator):
   ```bash
   winget install ffmpeg
   setx PATH "%PATH%;C:\Program Files\ffmpeg\bin"
   ```
3. Double-click `setup.bat` to:
   - Create virtual environment
   - Install dependencies
   - Download Swiss German Whisper model (approximately 3GB)
4. Double-click `run.bat` to start the application

### macOS/Linux

1. Install Python 3.8 or newer
2. Install ffmpeg:
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` or equivalent
3. Make scripts executable:
   ```bash
   chmod +x setup.sh run.sh
   ```
4. Run setup:
   ```bash
   ./setup.sh
   ```
   This will:
   - Create virtual environment
   - Install dependencies
   - Download Swiss German Whisper model (approximately 3GB)
5. Start application:
   ```bash
   ./run.sh
   ```

## Usage

The application works completely offline after initial setup. No internet connection is required for transcription.

2. Click "choose audio files" to select one or more audio/video files
3. Select desired output format(s)
4. Click "start" to begin transcription

The transcribed files will be saved in the same directory as the input files, with the same name but different extensions based on the selected output formats.

## Supported File Formats

### Audio
- MP3, M4A, M4B, M4P, FLAC
- OGG, OGA, MOGG, WAV, WMA
- MMF, AA, AAX

### Video
- MP4, M4V, MKV, WEBM
- AVI, MOV, WMV, FLV
- And many more video formats

## Output Formats

- XLSX: Excel spreadsheet with timestamps and text
- SRT: SubRip subtitle format
- TXT: Plain text
- VTT: WebVTT subtitle format
- TSV: Tab-separated values
- JSON: Structured data format

## Changes from Original Version

- Uses the specialized Swiss German Whisper Large V3 Turbo model
- Removed language selection (optimized for Swiss German)
- Improved code organization with separate modules
- Enhanced progress tracking
- Better error handling and resource management

## Credits

- Model: [Whisper Large V3 Turbo Swiss German](https://huggingface.co/nizarmichaud/whisper-large-v3-turbo-swissgerman)
- Original Whisper model: [OpenAI](https://github.com/openai/whisper)
