@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Verifying ffmpeg installation...
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
