@echo off

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists. Skipping creation.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip

rem Check if accelerate is already installed
pip show accelerate > nul 2>&1
if errorlevel 1 (
    echo Installing accelerate...
    pip install git+https://github.com/huggingface/accelerate
) else (
    echo accelerate is already installed. Skipping installation.
)

echo Installing other dependencies...
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

rem echo Verifying ffmpeg installation...
rem (will be displayed in verify_ffmpeg.py; else it will be shown twice)
python src\verify_ffmpeg.py
if errorlevel 1 (
    echo.
    echo Please install ffmpeg and run setup.bat again
    pause
    exit /b 1
)

echo Downloading Swiss German Whisper model...
python src\download_model.py

echo.
echo Setup complete! The application is ready to use:
echo 1. Run 'run.bat'
echo.
echo Press any key to exit...
pause >nul
