"""
GenAI Instagram Intelligence Studio - FFmpeg and FFprobe Diagnostics
Verifies FFmpeg and FFprobe binaries and multimedia processing capabilities.
"""

import sys
import shutil
import subprocess

def check_ffmpeg():
    print("=" * 60)
    print("FFmpeg and FFprobe Diagnostics")
    print("=" * 60)
    
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    
    if ffmpeg_bin:
        print(f"FFmpeg Binary: FOUND -> {ffmpeg_bin}")
        try:
            res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True, timeout=5)
            first_line = res.stdout.splitlines()[0] if res.stdout else "Unknown version"
            print(f"Version: {first_line}")
        except Exception as e:
            print(f"Error executing ffmpeg: {e}")
    else:
        print("FFmpeg Binary: NOT FOUND in system PATH")
        print("Tip: Install FFmpeg from https://ffmpeg.org/download.html and add to PATH.")

    print("-" * 60)
    if ffprobe_bin:
        print(f"FFprobe Binary: FOUND -> {ffprobe_bin}")
        try:
            res = subprocess.run([ffprobe_bin, "-version"], capture_output=True, text=True, timeout=5)
            first_line = res.stdout.splitlines()[0] if res.stdout else "Unknown version"
            print(f"Version: {first_line}")
        except Exception as e:
            print(f"Error executing ffprobe: {e}")
    else:
        print("FFprobe Binary: NOT FOUND in system PATH")
    print("=" * 60)

if __name__ == "__main__":
    check_ffmpeg()
