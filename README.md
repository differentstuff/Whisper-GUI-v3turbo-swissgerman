# Whisper GUI

An interface to streamline transcriptions using Whisper v3.
https://huggingface.co/nizarmichaud/whisper-large-v3-turbo-swissgerman

- transcribe video files up to ~ 25 MiB
- transcribe audio files of any size  
     *(Audio files larger than 25 MiB are split into segments with an overlap of 2 seconds. Finally the segment transcriptions are merged.)*

![Screenshot of Whisper GUI](./docs/whisper_gui_screenshot.png?raw=true)

## Requirements

- whisper (see requirements online)
- nicegui
- pywebview
- openpyxl
- pydub
- simpleaudio

## Developer

### Notes

| :snake: Python Version |
|:-|
| As of now (2024/10/19) Whisper works with Python **3.9** or **3.10**. Make sure to select correct version in your IDE (note to self: in VS Code select Python 3.9.0 ('.venv') in lower right corner). |

| :information_source: Virtual Environment |
|:-|
| Its best practice to setup a virtual environment for the project. Make sure to activate it (`.\.venv\Scripts\activate`) before installing libraries! |

### Setup Example

1. Create a virtual environment: .venv  
    Make sure to initialize the virtual environment using the correct python version (3.9 or 3.10 see notes):
   
       C:\Users\\[UserName]\AppData\Local\Programs\Python\Python39\python.exe -m venv .venv
   
3. Activate the virtual environment
   
       .venv\Scripts\activate
   
5. Install the following dependencies
   
       pip install git+https://github.com/openai/whisper.git  
       pip install blobfile  
       pip install nicegui  
       pip install pywebview
       pip install openpyxl
       pip install pydub
       pip install simpleaudio

### Build Executable

0. first time, install pyinstaller in python environment
    
       (.venv) pip install pyinstaller

1. run nicegui-pack

       (.venv) nicegui-pack --windowed --name "Whisper GUI" main.py

2. copy the following files and folders to `.\dist\Whisper GUI\_internal`
    - `sound_effect_finished.wav`
    - `.\.venv\Lib\site-packages\whisper`
    - `.\.venv\Lib\site-packages\tiktoken_ext`

## Whisper Offline Use

Add folder `C:\Users\[UserName]\.cache\whisper`. Move the files **docs/vocab.bpe** and **docs/encoder.json** to this folder. Update your local copy of openai_public.py. If you created a venv this file is located in `.venv\Lib\site-packages\tiktoken_ext\openai_public.py` otherwise it probably is in `C:\Users\[UserName]\AppData\Local\Programs\Python\Python310-32\Lib\site-packagespython3.9\site-packages\tiktoken_ext\openai_public.py`. Remove the URL "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/" and replace it with your local copy, e.g.:
            
        def gpt2():
        mergeable_ranks = data_gym_to_mergeable_bpe_ranks(  
            vocab_bpe_file="C:/Users/[Username]/.cache/whisper/vocab.bpe",   
            encoder_json_file="C:/Users/[Username]/.cache/whisper/encoder.json",  
        )
