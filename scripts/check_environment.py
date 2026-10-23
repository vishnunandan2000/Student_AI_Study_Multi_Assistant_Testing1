"""
GenAI Instagram Intelligence Studio & Autonomous Video Production Platform
System Environment & Dependency Validation Script
"""

import sys
import shutil
import platform
from pathlib import Path

# Ensure root in path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.core.moviepy_compat import get_moviepy_diagnostics


def check_environment():
    diag = get_moviepy_diagnostics()

    print("=" * 65)
    print("GenAI Video Studio - System Environment & Dependency Validation")
    print("=" * 65)
    print(f"Python Version:   {diag['python_version']} ({diag['python_bits']})")
    print(f"Operating System: {diag['os']}")
    print(f"MoviePy Version:  {diag['moviepy_version']} | Mode: {diag['selected_api_mode']}")
    print(f"FFmpeg Path:      {diag['ffmpeg_executable']}")
    print(f"FFmpeg Version:   {diag['ffmpeg_version']}")
    print(f"FFprobe Path:     {diag['ffprobe_executable']}")
    print("-" * 65)

    # Check key Python packages
    packages = [
        "gradio",
        "requests",
        "pydantic",
        "pandas",
        "plotly",
        "PIL",
        "reportlab",
        "moviepy",
        "cv2",
        "sqlite3",
        "dotenv",
        "imageio_ffmpeg",
        "faster_whisper"
    ]
    print("Python Library Availability:")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [x] {pkg:<18}: AVAILABLE ({ver})")
        except ImportError:
            print(f"  [ ] {pkg:<18}: NOT INSTALLED")

    print("-" * 65)
    status_str = "READY (Supported Configuration)" if diag["is_environment_supported"] else "INCOMPLETE"
    print(f"Studio Environment Status: {status_str}")
    print("=" * 65)


if __name__ == "__main__":
    check_environment()
