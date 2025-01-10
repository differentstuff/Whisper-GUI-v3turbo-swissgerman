import os
import sys
import subprocess
import shutil

def verify_ffmpeg():
    """Verify ffmpeg installation in user space"""
    print("\nVerifying ffmpeg installation...")
    
    # Check if ffmpeg is accessible
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"Found ffmpeg at: {ffmpeg_path}")
        return True
            
    print("ERROR: ffmpeg not found!")
    print("Please install ffmpeg:")
    if sys.platform == "win32":
        print("\n1. Open Command Prompt (not as Administrator)")
        print("2. Run: winget install ffmpeg")
        print("3. Close all Command Prompt windows")
        print("4. Start the application again")
    elif sys.platform == "darwin":
        print("\nRun: brew install ffmpeg")
    else:
        print("\nRun: sudo apt install ffmpeg")
    return False

if __name__ == "__main__":
    if not verify_ffmpeg():
        sys.exit(1)
