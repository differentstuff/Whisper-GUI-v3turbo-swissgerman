@echo off

:: go to root
cd ..

:: Setup virtual environment
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

:: Check accelerate installation
pip show accelerate > nul 2>&1
if errorlevel 1 (
    echo Installing accelerate...
    pip install git+https://github.com/huggingface/accelerate
) else (
    echo accelerate is already installed. Skipping installation.
)

echo Installing other dependencies...
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118


:: Verify ffmpeg
python install\s_verify_ffmpeg.py
if errorlevel 1 (
    echo.
    echo Please install ffmpeg and run setup.bat again
    pause
    exit /b 1
)


echo.
echo Setup complete! The application is ready to use:
echo 1. Run 'run.bat'
echo.
echo Press any key to exit...
pause >nul
