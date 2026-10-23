"""
Centralized MoviePy 1.x and 2.x Compatibility Layer
====================================================
Provides an explicit, centralized abstraction layer across MoviePy 1.x and MoviePy 2.x:
- Version detection and environment diagnostics.
- Unified import handling (moviepy vs moviepy.editor).
- Safe file loading (video and audio).
- Safe clipping, resizing, cropping, duration, audio, and concatenation helpers.
- Windows-safe video rendering with logfile redirection to avoid pipe buffer deadlocks.
- High-quality 1080x1920 (9:16) vertical reframing with full timeline duration preservation.
- Decoupled short-form (<60s) clipping vs full-length vertical video export.
- Media validation via FFprobe/FFmpeg.
"""

import os
import sys
import json
import shutil
import logging
import platform
import subprocess
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np

logger = logging.getLogger(__name__)

# =====================================================================
# 1. Version Detection & Core Imports
# =====================================================================

MOVIEPY_VERSION = "unknown"
MOVIEPY_MAJOR = 2

try:
    import moviepy
    MOVIEPY_VERSION = getattr(moviepy, "__version__", "unknown")
    try:
        MOVIEPY_MAJOR = int(MOVIEPY_VERSION.split(".")[0])
    except (ValueError, IndexError):
        MOVIEPY_MAJOR = 2
except ImportError:
    MOVIEPY_VERSION = "missing"
    MOVIEPY_MAJOR = 0

# Import clips based on detected MoviePy major version
if MOVIEPY_MAJOR == 1:
    try:
        from moviepy.editor import (
            VideoFileClip,
            AudioFileClip,
            CompositeVideoClip,
            concatenate_videoclips,
            ImageClip,
            ColorClip,
            VideoClip
        )
    except (ImportError, ModuleNotFoundError):
        from moviepy import (
            VideoFileClip,
            AudioFileClip,
            CompositeVideoClip,
            concatenate_videoclips,
            ImageClip,
            ColorClip,
            VideoClip
        )
else:
    # MoviePy 2.x+
    from moviepy import (
        VideoFileClip,
        AudioFileClip,
        CompositeVideoClip,
        concatenate_videoclips,
        ImageClip,
        ColorClip,
        VideoClip
    )

import imageio_ffmpeg


# =====================================================================
# 2. Comprehensive Diagnostics & Resolvers
# =====================================================================

def get_ffmpeg_binary_path() -> str:
    """Resolves the best available FFmpeg executable path."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def get_ffprobe_binary_path() -> Optional[str]:
    """Resolves system FFprobe executable path if available."""
    sys_ffprobe = shutil.which("ffprobe")
    if sys_ffprobe:
        return sys_ffprobe
    # Check beside imageio_ffmpeg binary
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cand = os.path.join(os.path.dirname(exe), "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    return None


def get_moviepy_diagnostics() -> Dict[str, Any]:
    """
    Returns complete runtime diagnostic metadata for MoviePy, Python,
    FFmpeg, and media processing components.
    """
    ffmpeg_exe = get_ffmpeg_binary_path()
    ffprobe_exe = get_ffprobe_binary_path()

    ffmpeg_ver = "unknown"
    if ffmpeg_exe and os.path.exists(ffmpeg_exe):
        try:
            res = subprocess.run([ffmpeg_exe, "-version"], capture_output=True, text=True, timeout=5)
            first_line = res.stdout.splitlines()[0] if res.stdout else "available"
            ffmpeg_ver = first_line
        except Exception as e:
            ffmpeg_ver = f"detected ({e})"

    api_mode = f"moviepy_v{MOVIEPY_MAJOR} ({MOVIEPY_VERSION})"

    return {
        "python_version": platform.python_version(),
        "python_bits": platform.architecture()[0],
        "os": f"{platform.system()} {platform.release()}",
        "moviepy_version": MOVIEPY_VERSION,
        "moviepy_major": MOVIEPY_MAJOR,
        "selected_api_mode": api_mode,
        "ffmpeg_executable": ffmpeg_exe,
        "ffmpeg_version": ffmpeg_ver,
        "ffprobe_executable": ffprobe_exe or "MISSING (system fallback will use FFmpeg/OpenCV)",
        "is_environment_supported": (MOVIEPY_MAJOR in (1, 2) and os.path.exists(ffmpeg_exe))
    }


def log_startup_diagnostics():
    """Logs runtime diagnostic banner on application initialization."""
    diag = get_moviepy_diagnostics()
    logger.info("=" * 65)
    logger.info("MEDIA PROCESSING & MOVIEPY COMPATIBILITY DIAGNOSTICS")
    logger.info("=" * 65)
    logger.info(f"Python:       {diag['python_version']} ({diag['python_bits']}) on {diag['os']}")
    logger.info(f"MoviePy:      {diag['moviepy_version']} (API Mode: {diag['selected_api_mode']})")
    logger.info(f"FFmpeg Exe:   {diag['ffmpeg_executable']}")
    logger.info(f"FFprobe Exe:  {diag['ffprobe_executable']}")
    logger.info(f"Environment:  {'SUPPORTED & READY' if diag['is_environment_supported'] else 'WARNING - INCOMPLETE'}")
    logger.info("=" * 65)


# =====================================================================
# 3. File Loading & Resource Management
# =====================================================================

def load_videofile_safe(video_path: str, **kwargs) -> Any:
    """
    Safely loads a video file with verification of existence and readability.
    """
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")
    if os.path.getsize(video_path) == 0:
        raise ValueError(f"Video file is empty (0 bytes): {video_path}")
    return VideoFileClip(video_path, **kwargs)


def load_audiofile_safe(audio_path: str, **kwargs) -> Any:
    """
    Safely loads an audio file with verification of existence and readability.
    """
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
    if os.path.getsize(audio_path) == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {audio_path}")
    return AudioFileClip(audio_path, **kwargs)


def close_clip_resources(*clips: Any):
    """
    Closes video clips and their attached audio readers to release open file handles.
    """
    for c in clips:
        if c is None:
            continue
        try:
            if hasattr(c, "audio") and c.audio is not None:
                try:
                    c.audio.close()
                except Exception:
                    pass
            if hasattr(c, "close"):
                c.close()
        except Exception:
            pass


# =====================================================================
# 4. Centralized Clip Operation Wrappers (MoviePy 1.x & 2.x Safe)
# =====================================================================

def clip_time(clip: Any, start_t: float, end_t: float) -> Any:
    """
    Safely extracts a time subclip across MoviePy 1.x (.subclip) and MoviePy 2.x (.subclipped).
    Guarantees no AttributeError: 'VideoFileClip' object has no attribute 'subclip'.
    """
    s = max(0.0, float(start_t))
    e = max(s, float(end_t))
    if hasattr(clip, "duration") and clip.duration is not None:
        e = min(float(clip.duration), e)

    if hasattr(clip, "subclipped"):
        return clip.subclipped(s, e)
    elif hasattr(clip, "subclip"):
        return clip.subclip(s, e)
    else:
        raise AttributeError(f"Clip type '{type(clip).__name__}' does not support subclipped/subclip.")


def resize_clip(clip: Any, width: Optional[int] = None, height: Optional[int] = None, newsize: Optional[Tuple[int, int]] = None) -> Any:
    """
    Safely resizes video clips across MoviePy 1.x (.resize) and MoviePy 2.x (.resized).
    """
    kwargs = {}
    if newsize is not None:
        kwargs["new_size"] = newsize
    else:
        if width is not None:
            kwargs["width"] = int(width)
        if height is not None:
            kwargs["height"] = int(height)

    # In MoviePy 2.x: .resized(...)
    if hasattr(clip, "resized"):
        return clip.resized(**kwargs)
    elif hasattr(clip, "resize"):
        if "new_size" in kwargs:
            kwargs["newsize"] = kwargs.pop("new_size")
        return clip.resize(**kwargs)
    else:
        raise AttributeError(f"Clip type '{type(clip).__name__}' does not support resized/resize.")


def crop_clip(
    clip: Any,
    x1: Optional[int] = None,
    y1: Optional[int] = None,
    x2: Optional[int] = None,
    y2: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    x_center: Optional[int] = None,
    y_center: Optional[int] = None
) -> Any:
    """
    Safely crops video clips across MoviePy 1.x (.crop(...)) and MoviePy 2.x (with_effects([vfx.Crop(...)])).
    """
    kwargs = {}
    if x1 is not None: kwargs["x1"] = int(x1)
    if y1 is not None: kwargs["y1"] = int(y1)
    if x2 is not None: kwargs["x2"] = int(x2)
    if y2 is not None: kwargs["y2"] = int(y2)
    if width is not None: kwargs["width"] = int(width)
    if height is not None: kwargs["height"] = int(height)
    if x_center is not None: kwargs["x_center"] = int(x_center)
    if y_center is not None: kwargs["y_center"] = int(y_center)

    if hasattr(clip, "crop"):
        return clip.crop(**kwargs)
    elif hasattr(clip, "cropped"):
        return clip.cropped(**kwargs)
    elif hasattr(clip, "with_effects"):
        try:
            import moviepy.video.fx as vfx
            return clip.with_effects([vfx.Crop(**kwargs)])
        except Exception as e:
            logger.warning(f"vfx.Crop effect notice: {e}")
    raise AttributeError(f"Clip type '{type(clip).__name__}' does not support crop.")


def center_crop_to_aspect(clip: Any, target_w: int = 1080, target_h: int = 1920) -> Any:
    """
    Scales and center-crops a clip to strictly match target_w x target_h (e.g. 9:16 aspect ratio).
    """
    orig_w, orig_h = clip.w, clip.h
    scale = max(target_w / orig_w, target_h / orig_h)
    scaled_w = int(round(orig_w * scale))
    scaled_h = int(round(orig_h * scale))

    scaled = resize_clip(clip, width=scaled_w, height=scaled_h)
    x_c = scaled_w // 2
    y_c = scaled_h // 2
    return crop_clip(scaled, x_center=x_c, y_center=y_c, width=target_w, height=target_h)


def set_clip_duration(clip: Any, duration: float) -> Any:
    """
    Safely sets clip duration across MoviePy 1.x (.set_duration) and MoviePy 2.x (.with_duration).
    """
    dur = max(0.0, float(duration))
    if hasattr(clip, "with_duration"):
        return clip.with_duration(dur)
    elif hasattr(clip, "set_duration"):
        return clip.set_duration(dur)
    return clip


def set_clip_position(clip: Any, pos: Union[str, Tuple[Union[str, int, float], Union[str, int, float]]]) -> Any:
    """
    Safely sets clip positioning across MoviePy 1.x (.set_position) and MoviePy 2.x (.with_position).
    """
    if hasattr(clip, "with_position"):
        return clip.with_position(pos)
    elif hasattr(clip, "set_position"):
        return clip.set_position(pos)
    return clip


def set_clip_audio(video_clip: Any, audio_clip: Any) -> Any:
    """
    Safely attaches an audio clip to a video clip across MoviePy 1.x (.set_audio) and 2.x (.with_audio).
    """
    if hasattr(video_clip, "with_audio"):
        return video_clip.with_audio(audio_clip)
    elif hasattr(video_clip, "set_audio"):
        return video_clip.set_audio(audio_clip)
    return video_clip


def concatenate_clips(clips: List[Any], method: str = "compose") -> Any:
    """
    Safely concatenates multiple video subclips.
    """
    if not clips:
        raise ValueError("Cannot concatenate empty list of clips.")
    if len(clips) == 1:
        return clips[0]
    return concatenate_videoclips(clips, method=method)


def export_videofile_safe(
    clip: Any,
    output_path: str,
    fps: Optional[float] = None,
    codec: str = "libx264",
    audio_codec: str = "aac",
    preset: str = "ultrafast",
    threads: int = 4,
    write_logfile: bool = True,
    logger_opt: Optional[str] = None
) -> str:
    """
    Exports video to MP4 with explicit protection against Windows stderr pipe buffer deadlocks.
    By setting write_logfile=True, FFmpeg writes encoding telemetry to a disk log instead of
    blocking on unconsumed stderr pipe buffers during long video encodes.
    """
    out_p = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    target_fps = fps or getattr(clip, "fps", 25.0) or 25.0

    if not hasattr(clip, "duration") or clip.duration is None:
        raise ValueError("Cannot export clip without a defined duration.")

    clip.write_videofile(
        out_p,
        fps=target_fps,
        codec=codec,
        audio_codec=audio_codec,
        preset=preset,
        threads=threads,
        write_logfile=write_logfile,
        logger=logger_opt
    )

    if not os.path.exists(out_p) or os.path.getsize(out_p) == 0:
        raise RuntimeError(f"Video export failed to create valid file at: {out_p}")

    log_file = out_p + ".log"
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except Exception:
            pass

    return out_p


# =====================================================================
# 5. Media Validation via FFprobe / FFmpeg
# =====================================================================

def validate_media_file_ffprobe(video_path: str) -> Dict[str, Any]:
    """
    Validates container integrity, resolution, frame rate, duration, and audio stream
    using FFprobe when available, falling back cleanly to FFmpeg inspection.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Media file does not exist: {video_path}")

    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise ValueError(f"Media file is empty (0 bytes): {video_path}")

    ffprobe_exe = get_ffprobe_binary_path()
    if ffprobe_exe:
        cmd = [
            ffprobe_exe,
            "-v", "error",
            "-show_entries", "stream=codec_type,width,height,r_frame_rate,duration:format=duration,size",
            "-of", "json",
            video_path
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                streams = data.get("streams", [])
                format_info = data.get("format", {})
                v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
                has_audio = any(s.get("codec_type") == "audio" for s in streams)

                duration = float(format_info.get("duration") or v_stream.get("duration") or 0.0)
                width = int(v_stream.get("width", 0))
                height = int(v_stream.get("height", 0))

                return {
                    "valid": True,
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "has_audio": has_audio,
                    "file_size": file_size,
                    "probe_engine": "ffprobe"
                }
        except Exception as e:
            logger.warning(f"FFprobe execution notice ({e}), using FFmpeg fallback.")

    # Fallback to inspecting with FFmpeg or VideoFileClip
    ffmpeg_exe = get_ffmpeg_binary_path()
    try:
        res = subprocess.run([ffmpeg_exe, "-i", video_path], capture_output=True, text=True, timeout=10)
        output = res.stderr
        has_audio = "Audio:" in output
        # Read duration and resolution using VideoFileClip
        with VideoFileClip(video_path) as c:
            return {
                "valid": True,
                "width": c.w,
                "height": c.h,
                "duration": float(c.duration or 0.0),
                "has_audio": has_audio,
                "file_size": file_size,
                "probe_engine": "ffmpeg/moviepy"
            }
    except Exception as e:
        raise RuntimeError(f"Failed to validate media file integrity: {e}")


# =====================================================================
# 6. Universal Full-Length Vertical (9:16) Reframing Engine
# =====================================================================

def reframe_to_vertical_9_16(
    video_path: str,
    output_path: str,
    target_w: int = 1080,
    target_h: int = 1920,
    fps: float = 25.0,
    threads: int = 4,
    crop_strategy: str = "fit"
) -> str:
    """
    Converts any landscape or arbitrary video into a pristine 1080x1920 (9:16) vertical video:
    - PRESERVES THE COMPLETE TIMELINE DURATION (no arbitrary 60s cutoff).
    - If crop_strategy == 'crop': Scaled and center-cropped to strictly fill 1080x1920.
    - If crop_strategy == 'fit': Crisp centered player with ambient dark slate margins (1080x1920).
    - Preserves complete audio track and perfect AV synchronization.
    - Windows-safe video rendering.
    """
    target_p = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(target_p), exist_ok=True)

    with load_videofile_safe(video_path) as orig_clip:
        total_duration = float(orig_clip.duration)
        if total_duration <= 0:
            raise ValueError(f"Invalid video duration: {total_duration}s")

        if crop_strategy == "crop":
            # Center crop to 9:16 aspect ratio
            final_clip = center_crop_to_aspect(orig_clip, target_w=target_w, target_h=target_h)
            if orig_clip.audio is not None:
                final_clip = set_clip_audio(final_clip, orig_clip.audio)
        else:
            # Fit width (1080px) and center on 1080x1920 slate canvas
            centered_clip = resize_clip(orig_clip, width=target_w)
            c_w, c_h = centered_clip.w, centered_clip.h
            y_center = max(0, (target_h - c_h) // 2)
            pos_clip = set_clip_position(centered_clip, ("center", y_center))
            bg_clip = set_clip_duration(ColorClip(size=(target_w, target_h), color=(15, 23, 42)), total_duration)

            composite = CompositeVideoClip([bg_clip, pos_clip], size=(target_w, target_h))
            if orig_clip.audio is not None:
                composite = set_clip_audio(composite, orig_clip.audio)
            final_clip = composite

        # Render full duration without truncation
        export_videofile_safe(
            final_clip,
            target_p,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=threads,
            write_logfile=True,
            logger_opt=None
        )

        close_clip_resources(final_clip)

    return target_p


# =====================================================================
# 7. Dedicated Platform Short-Form Extractor (< 60s Excerpt)
# =====================================================================

def extract_platform_short_clip(
    video_path: str,
    output_path: str,
    max_duration: float = 58.0,
    start_offset: float = 0.0,
    fps: float = 25.0
) -> str:
    """
    Separately extracts an excerpt specifically tailored for platform short-form constraints (<=60s).
    This function is strictly decoupled from the full-length vertical export.
    """
    target_p = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(target_p), exist_ok=True)

    with load_videofile_safe(video_path) as full_clip:
        total_dur = float(full_clip.duration)
        start_t = min(max(0.0, start_offset), max(0.0, total_dur - 1.0))
        sub_dur = min(total_dur - start_t, max(5.0, max_duration))
        end_t = start_t + sub_dur

        # Extract excerpt
        excerpt = clip_time(full_clip, start_t, end_t)

        target_w, target_h = 1080, 1920
        centered_clip = resize_clip(excerpt, width=target_w)
        y_center = max(0, (target_h - centered_clip.h) // 2)
        pos_clip = set_clip_position(centered_clip, ("center", y_center))

        bg_clip = set_clip_duration(ColorClip(size=(target_w, target_h), color=(15, 23, 42)), sub_dur)
        composite = CompositeVideoClip([bg_clip, pos_clip], size=(target_w, target_h))
        if excerpt.audio is not None:
            composite = set_clip_audio(composite, excerpt.audio)

        export_videofile_safe(
            composite,
            target_p,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=4,
            write_logfile=True,
            logger_opt=None
        )

        close_clip_resources(composite, bg_clip, pos_clip, centered_clip, excerpt)

    return target_p
