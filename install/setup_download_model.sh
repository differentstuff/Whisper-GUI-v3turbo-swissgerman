#!/bin/bash

MODEL_ID="nizarmichaud/whisper-large-v3-turbo-swissgerman"

# go to root
cd ..

if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run setup.sh first to create the virtual environment."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Verifying model: $MODEL_ID"
python install/s_download_model.py "$MODEL_ID" > /dev/null

echo
echo "Model setup complete!"
echo "You can now run the application using './run.sh'"
echo
read -p "Press Enter to exit..."