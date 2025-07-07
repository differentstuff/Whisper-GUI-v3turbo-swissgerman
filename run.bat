@echo off
echo Starting application Server... Please wait
call venv\Scripts\activate.bat
set PYTHONWARNINGS=ignore
python main.py
