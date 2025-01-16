#!/bin/bash

# go to root
cd ..

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists. Skipping creation."
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
python -m pip install --upgrade pip

# Check if accelerate is already installed
if ! pip show accelerate > /dev/null 2>&1; then
    echo "Installing accelerate..."
    pip install git+https://github.com/huggingface/accelerate
else
    echo "accelerate is already installed. Skipping installation."
fi

echo "Installing other dependencies..."
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

# Verify ffmpeg installation
python s_verify_ffmpeg.py
if [ $? -ne 0 ]; then
    echo
    echo "Please install ffmpeg and run setup.sh again"
    read -p "Press Enter to exit..."
    exit 1
fi

echo
echo "Setup complete! The application is ready to use:"
echo "1. Run './run.sh'"
echo
read -p "Press Enter to exit..."