# Whisper Interface v2

A GUI application for transcribing audio/video files using the specialized **Whisper Large V3 Turbo** model.

## Features

- **User-friendly GUI**: Simple and intuitive interface
- **GPU acceleration**: Faster performance with GPU, fallback to CPU if unavailable
- **Wide format support**: Works with multiple audio and video formats
- **Automatic file splitting**: Handles large files seamlessly
- **Multiple output formats**: Export transcriptions in XLSX, SRT, TXT, VTT, TSV, or JSON
- **Timestamps toggle**: Option to include or exclude timestamps in exports (enabled by default)
- **Dark/Light mode**: Choose your preferred theme
- **Progress tracking**: Real-time progress updates with sound notifications
- **100% offline processing**: No internet required after model download
- **Color-coded progress**: Easy-to-read progress updates in the terminal
- **Detailed logging**: Logs saved locally in the `/logs` directory

## Installation & Usage

### Windows

1. Install **Python 3.8 or newer** from [python.org](https://www.python.org/downloads/).
2. Install **ffmpeg** by running the following commands in Command Prompt (as Administrator):
   ```bash
   winget install ffmpeg
   setx PATH "%PATH%;C:\Program Files\ffmpeg\bin"
   ```
3. Double-click `install\setup.bat` to:
   - Create a virtual environment
   - Install all required dependencies
4. Double-click `install\setup_download_model.bat` to:
   - Download the **Swiss German Whisper model** (approximately 3GB)
5. Double-click `run.bat` to start the application.

### macOS/Linux

1. Install **Python 3.8 or newer**.
2. Install **ffmpeg**:
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` (or equivalent for your distribution)
3. Make the scripts executable:
   ```bash
   chmod +x install/setup.sh run.sh
   ```
4. Run the setup script:
   ```bash
   ./install/setup.sh
   ```
   This will:
   - Create a virtual environment
   - Install all required dependencies
5. Download the model:
   ```bash
   ./install/setup_download_model.sh
   ```
   - This will download the **Swiss German Whisper model** (approximately 3GB).
6. Start the application:
   ```bash
   ./run.sh
   ```

## Changing the Model

The application uses the **Swiss German Whisper Large V3 Turbo** model by default. If you want to use a different Whisper model, you can modify the hardcoded model ID in the `setup_download_model.bat` (Windows) or `setup_download_model.sh` (Linux/macOS) file.

- **Windows**: Open `install\setup_download_model.bat` and change the `MODEL_ID` value:
  ```batch
  set MODEL_ID="your-new-model-id"
  ```
- **Linux/macOS**: Open `install/setup_download_model.sh` and change the `MODEL_ID` value:
  ```bash
  MODEL_ID="your-new-model-id"
  ```

You can find other Whisper models on [Hugging Face](https://huggingface.co/models).

## Usage

The application works completely offline after the initial setup. No internet connection is required for transcription.

1. Launch the application by running `run.bat` (Windows) or `./run.sh` (Linux/macOS).
2. Click **"Choose audio files"** to select one or more audio/video files.
3. Use the "Include Timestamps" checkbox to toggle timestamps in the exported files (enabled by default).
4. Select the desired output format(s).
5. Click **"Start"** to begin transcription.

The transcribed files will be saved in the same directory as the input files, with the same name but different extensions based on the selected output formats.

## Configuring Audio Settings

You can configure several audio-related settings in the file src/audio_config.py. 
These settings are explained within that file. Please refer to it for further customization options.

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

- **XLSX**: Excel spreadsheet with timestamps and text
- **SRT**: SubRip subtitle format
- **TXT**: Plain text
- **VTT**: WebVTT subtitle format
- **TSV**: Tab-separated values
- **JSON**: Structured data format

## Changes from Original Version

- Runs the model on GPU, with fallback to CPU
- Uses the specialized **Swiss German Whisper Large V3 Turbo** model
- Improved code organization with separate modules
- Enhanced progress tracking
- Better error handling and resource management
- Added a GUI checkbox to toggle timestamps in exports (enabled by default)

## Credits

- **Model**: [Whisper Large V3 Turbo Swiss German](https://huggingface.co/nizarmichaud/whisper-large-v3-turbo-swissgerman)
- **Original Whisper model**: [OpenAI](https://github.com/openai/whisper)
- **Whisper Interface v1**: [soer1i](https://github.com/soer1i/Whisper-GUI/tree/main)

## Known Bugs

- Button 'Abort' doesn't work
- App looses connection after first transcription is finished
- App has no clean Exit Button