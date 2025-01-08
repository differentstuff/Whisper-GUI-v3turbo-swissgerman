#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Verifying ffmpeg installation..."
python src/verify_ffmpeg.py
if [ $? -ne 0 ]; then
    echo
    echo "Please install ffmpeg and run setup.sh again"
    read -n 1 -s -r -p "Press any key to exit..."
    echo
    exit 1
fi

echo "Downloading Swiss German Whisper model..."
python src/download_model.py

echo
echo "Setup complete! The application is ready to use:"
echo "1. Run './run.sh'"
echo
read -n 1 -s -r -p "Press any key to exit..."
echo
