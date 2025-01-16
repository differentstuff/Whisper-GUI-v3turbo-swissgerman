@echo off
setlocal EnableDelayedExpansion

set MODEL_ID="nizarmichaud/whisper-large-v3-turbo-swissgerman"

:: Go to root (parent directory)
cd ..

:: Check if virtual environment exists
if not exist venv (
    echo Error: Virtual environment not found!
    echo Please run setup.bat first to create the virtual environment.
    pause >nul
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Error: Failed to activate virtual environment.
    pause >nul
    exit /b 1
)

:: Download/verify Whisper model - note the path to the Python script
echo Downloading/Verifying model: %MODEL_ID%
python install\s_download_model.py %MODEL_ID% >nul
if errorlevel 1 (
    echo.
    echo Error: Model download failed. Please check your internet connection and try again.
    pause >nul
    exit /b 1
)

echo.
echo Model setup complete!
echo You can now run the application using 'run.bat'
echo.
echo Press any key to exit...
pause >nul