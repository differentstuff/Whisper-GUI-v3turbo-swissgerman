import os
import sys
import subprocess
from pydub.utils import which

def verify_ffmpeg():
    print("Verifying ffmpeg installation...")
    
    # Check if ffmpeg is in PATH
    ffmpeg_path = which("ffmpeg")
    if ffmpeg_path:
        print(f"Found ffmpeg at: {ffmpeg_path}")
        return True
        
    # Try common installation locations
    common_paths = [
        r"C:\Program Files\ffmpeg\bin",
        r"C:\ffmpeg\bin",
        os.path.expanduser("~/ffmpeg/bin"),
        "/usr/local/bin",
        "/usr/bin"
    ]
    
    for path in common_paths:
        if os.path.exists(os.path.join(path, "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")):
            # Add to PATH
            if sys.platform == "win32":
                os.environ["PATH"] = f"{path};{os.environ['PATH']}"
                # Also set permanent PATH using setx
                subprocess.run(["setx", "PATH", f"%PATH%;{path}"], check=True)
            else:
                os.environ["PATH"] = f"{path}:{os.environ['PATH']}"
            print(f"Added ffmpeg to PATH: {path}")
            return True
            
    print("ERROR: ffmpeg not found!")
    print("\nPlease install ffmpeg:")
    if sys.platform == "win32":
        print("1. Run as User: winget install ffmpeg")
        print("2. Restart your terminal")
    elif sys.platform == "darwin":
        print("Run: brew install ffmpeg")
    else:
        print("Run: sudo apt install ffmpeg")
    return False

if __name__ == "__main__":
    if not verify_ffmpeg():
        sys.exit(1)
