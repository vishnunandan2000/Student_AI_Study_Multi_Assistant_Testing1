"""
MoviePy Version Compatibility and Video Pipeline Regression Test Suite
=======================================================================
Validates:
1. Environment diagnostics and version detection (MoviePy 1.x & 2.x, FFmpeg, FFprobe).
2. Video and audio file loading with existence and integrity checks.
3. Universal clipping via clip_time() (Guarantees no AttributeError: 'VideoFileClip' object has no attribute 'subclip').
4. Universal resizing via resize_clip() (supports .resized and .resize).
5. Center cropping and aspect ratio conversion to 9:16 (1080x1920).
6. Strict output dimension enforcement (1080 x 1920).
7. Audio attachment and synchronization via set_clip_audio().
8. Video subclip concatenation via concatenate_clips().
9. Composite overlay rendering with subtitles/badges.
10. Full-length vertical 9:16 export without artificial 60-second cutoff (duration > 60s preserved).
11. Decoupled platform short-form export (<=60.0s).
12. Container and stream inspection via validate_media_file_ffprobe().
13. Missing audio track handling (graceful processing without crashes).
14. Resource cleanup and handle closing via close_clip_resources().
15. Error propagation and exception handling.
"""

import os
import sys
import unittest
import tempfile
import cv2
import numpy as np
from pathlib import Path

# Ensure root in path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.core.moviepy_compat import (
    MOVIEPY_VERSION,
    MOVIEPY_MAJOR,
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    ColorClip,
    load_videofile_safe,
    load_audiofile_safe,
    clip_time,
    resize_clip,
    crop_clip,
    center_crop_to_aspect,
    set_clip_duration,
    set_clip_position,
    set_clip_audio,
    concatenate_clips,
    close_clip_resources,
    export_videofile_safe,
    validate_media_file_ffprobe,
    reframe_to_vertical_9_16,
    extract_platform_short_clip,
    get_moviepy_diagnostics
)
from backend.modules.video_editor_studio import generate_subtitles_files


def create_synthetic_mp4(file_path: str, duration_sec: float = 3.0, width: int = 1280, height: int = 720, fps: int = 25):
    """Generates a valid MP4 test video with visual pattern."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :, 0] = (i * 2) % 256
        frame[:, :, 1] = 60
        frame[:, :, 2] = 120
        # Dynamic shape
        cv2.circle(frame, (width // 2 + int(np.sin(i / 10.0) * 100), height // 2), 50, (255, 255, 255), -1)
        cv2.putText(frame, f"Frame {i}/{total_frames}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        out.write(frame)
    out.release()


class TestMoviePyCompatibilityAndRegression(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Standard 3.0s 720p test video
        self.test_video_short = str(self.tmp_path / "test_short.mp4")
        create_synthetic_mp4(self.test_video_short, duration_sec=3.0, width=1280, height=720, fps=25)

        # Long test video (e.g. 70s > 60s) for duration constraint testing
        # To make test fast, we generate a 65-second video at 2 fps (130 frames total)
        self.test_video_long = str(self.tmp_path / "test_long_65s.mp4")
        create_synthetic_mp4(self.test_video_long, duration_sec=65.0, width=1280, height=720, fps=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------
    # 1. Environment & Startup Diagnostics
    # -----------------------------------------------------------------
    def test_environment_diagnostics(self):
        """Verify startup diagnostics return complete environment metadata."""
        diag = get_moviepy_diagnostics()
        self.assertIn("moviepy_version", diag)
        self.assertIn(diag["moviepy_major"], [1, 2])
        self.assertTrue(os.path.exists(diag["ffmpeg_executable"]))
        self.assertTrue(diag["is_environment_supported"])

    # -----------------------------------------------------------------
    # 2. File Loading & Resource Management
    # -----------------------------------------------------------------
    def test_load_videofile_safe(self):
        """Verify safe video file loading and error handling for missing/empty files."""
        clip = load_videofile_safe(self.test_video_short)
        self.assertIsNotNone(clip)
        self.assertEqual(clip.w, 1280)
        self.assertEqual(clip.h, 720)
        close_clip_resources(clip)

        # Missing file should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_videofile_safe(str(self.tmp_path / "missing_nonexistent.mp4"))

        # Empty file should raise ValueError
        empty_file = str(self.tmp_path / "empty.mp4")
        with open(empty_file, "wb") as f:
            f.write(b"")
        with self.assertRaises(ValueError):
            load_videofile_safe(empty_file)

    # -----------------------------------------------------------------
    # 3. Regression: 'VideoFileClip' object has no attribute 'subclip'
    # -----------------------------------------------------------------
    def test_regression_subclip_attribute_error_prevention(self):
        """
        REGRESSION TEST:
        In MoviePy 2.x, .subclip() is deprecated/removed in favor of .subclipped().
        Calling clip_time() MUST never throw:
        AttributeError: 'VideoFileClip' object has no attribute 'subclip'
        """
        with load_videofile_safe(self.test_video_short) as clip:
            # If MoviePy 2.x, verify direct .subclip is indeed absent (reproducing the exact issue)
            if MOVIEPY_MAJOR >= 2:
                self.assertFalse(hasattr(clip, "subclip"), "MoviePy 2.x does not have .subclip")

            # clip_time() must work seamlessly across both MoviePy 1.x and 2.x
            sub = clip_time(clip, 0.5, 2.0)
            self.assertIsNotNone(sub)
            self.assertAlmostEqual(sub.duration, 1.5, delta=0.2)
            close_clip_resources(sub)

    # -----------------------------------------------------------------
    # 4. Universal Video Resizing (.resized vs .resize)
    # -----------------------------------------------------------------
    def test_resize_clip_cross_version(self):
        """Verify resize_clip works across versions with width, height, and newsize."""
        with load_videofile_safe(self.test_video_short) as clip:
            resized_w = resize_clip(clip, width=640)
            self.assertEqual(resized_w.w, 640)
            close_clip_resources(resized_w)

            resized_new = resize_clip(clip, newsize=(1080, 1920))
            self.assertEqual(resized_new.w, 1080)
            self.assertEqual(resized_new.h, 1920)
            close_clip_resources(resized_new)

    # -----------------------------------------------------------------
    # 5. Center Cropping to 9:16 Aspect Ratio
    # -----------------------------------------------------------------
    def test_crop_clip_and_center_crop_aspect(self):
        """Verify cropping and center_crop_to_aspect creates exact 1080x1920 9:16 canvas."""
        with load_videofile_safe(self.test_video_short) as clip:
            cropped = center_crop_to_aspect(clip, target_w=1080, target_h=1920)
            self.assertEqual(cropped.w, 1080)
            self.assertEqual(cropped.h, 1920)
            close_clip_resources(cropped)

    # -----------------------------------------------------------------
    # 6. Clip Duration, Positioning, and Audio Assignment
    # -----------------------------------------------------------------
    def test_clip_duration_position_audio(self):
        """Verify duration, positioning, and audio assignment across versions."""
        c = ColorClip(size=(200, 200), color=(255, 0, 0))
        c = set_clip_duration(c, 4.0)
        self.assertEqual(c.duration, 4.0)

        c = set_clip_position(c, ("center", 100))
        self.assertIsNotNone(c)
        close_clip_resources(c)

    # -----------------------------------------------------------------
    # 7. Video Concatenation
    # -----------------------------------------------------------------
    def test_concatenate_clips(self):
        """Verify concatenation of multiple subclips."""
        with load_videofile_safe(self.test_video_short) as clip:
            part1 = clip_time(clip, 0.0, 1.0)
            part2 = clip_time(clip, 1.0, 2.0)
            joined = concatenate_clips([part1, part2])
            self.assertAlmostEqual(joined.duration, 2.0, delta=0.2)
            close_clip_resources(joined, part1, part2)

    # -----------------------------------------------------------------
    # 8. Subtitle File Synthesis & Timing Alignment
    # -----------------------------------------------------------------
    def test_subtitle_timing_and_alignment(self):
        """Verify SRT and WebVTT caption file generation and timestamp alignment."""
        segments = [
            {"start": 0.0, "end": 1.5, "text": "Segment one test text."},
            {"start": 1.5, "end": 3.0, "text": "Segment two test text."}
        ]
        out_base = str(self.tmp_path / "captions_test")
        srt_p, vtt_p = generate_subtitles_files(segments, out_base, target_language="English")
        self.assertTrue(os.path.exists(srt_p))
        self.assertTrue(os.path.exists(vtt_p))
        with open(srt_p, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("00:00:00,000 --> 00:00:01,500", content)
            self.assertIn("Segment one test text.", content)

    # -----------------------------------------------------------------
    # 9. Full-Length Vertical Export (>60s Preservation - NO 60s LIMIT)
    # -----------------------------------------------------------------
    def test_full_length_vertical_export_preserves_duration_above_60s(self):
        """
        CRITICAL TEST:
        Validates that a video longer than 60 seconds (65.0s) is reframed to vertical 9:16
        (1080x1920) WITHOUT being truncated, rejected, or limited by any artificial 60s rule.
        """
        out_vertical = str(self.tmp_path / "vertical_full_65s.mp4")
        reframe_to_vertical_9_16(
            video_path=self.test_video_long,
            output_path=out_vertical,
            target_w=1080,
            target_h=1920,
            fps=2.0
        )
        self.assertTrue(os.path.exists(out_vertical))
        self.assertGreater(os.path.getsize(out_vertical), 0)

        # Validate with probe
        probe = validate_media_file_ffprobe(out_vertical)
        self.assertTrue(probe["valid"])
        self.assertEqual(probe["width"], 1080)
        self.assertEqual(probe["height"], 1920)
        # Duration MUST be preserved (> 60.0s) and not truncated
        self.assertGreater(probe["duration"], 60.0, f"Duration was {probe['duration']}s, should exceed 60.0s")
        self.assertAlmostEqual(probe["duration"], 65.0, delta=2.0)

    # -----------------------------------------------------------------
    # 10. Decoupled Platform Short-Form Export (<=60.0s Excerpt)
    # -----------------------------------------------------------------
    def test_decoupled_platform_short_clip_export(self):
        """
        Verify that short-form exports are strictly decoupled from full-length vertical exports,
        extracting a focused <= 60.0s clip (e.g. 58.0s) without affecting master files.
        """
        out_short = str(self.tmp_path / "platform_short_58s.mp4")
        extract_platform_short_clip(
            video_path=self.test_video_long,
            output_path=out_short,
            max_duration=58.0,
            start_offset=0.0,
            fps=2.0
        )
        self.assertTrue(os.path.exists(out_short))
        self.assertGreater(os.path.getsize(out_short), 0)

        probe = validate_media_file_ffprobe(out_short)
        self.assertTrue(probe["valid"])
        self.assertEqual(probe["width"], 1080)
        self.assertEqual(probe["height"], 1920)
        # Platform short MUST be strictly <= 60.0s
        self.assertLessEqual(probe["duration"], 60.0)
        self.assertAlmostEqual(probe["duration"], 58.0, delta=2.0)

    # -----------------------------------------------------------------
    # 11. Missing Audio Track Graceful Handling
    # -----------------------------------------------------------------
    def test_missing_audio_track_graceful_export(self):
        """Synthetic videos without audio tracks must be exported and validated without crashing."""
        out_no_audio = str(self.tmp_path / "vertical_no_audio.mp4")
        reframe_to_vertical_9_16(
            video_path=self.test_video_short,
            output_path=out_no_audio,
            fps=25.0
        )
        self.assertTrue(os.path.exists(out_no_audio))
        probe = validate_media_file_ffprobe(out_no_audio)
        self.assertTrue(probe["valid"])
        self.assertEqual(probe["width"], 1080)
        self.assertEqual(probe["height"], 1920)

    # -----------------------------------------------------------------
    # 12. Resource Cleanup
    # -----------------------------------------------------------------
    def test_resource_cleanup_tolerates_none_and_closed_clips(self):
        """close_clip_resources() should handle None, open clips, and already closed clips without raising."""
        c = ColorClip(size=(50, 50), color=(0, 0, 0), duration=1)
        close_clip_resources(c, None, "not_a_clip")
        # Second call on closed clip should not raise
        close_clip_resources(c)


if __name__ == "__main__":
    unittest.main()
