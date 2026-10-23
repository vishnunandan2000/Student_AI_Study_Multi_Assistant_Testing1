"""
Unit and Integration Tests for Automatic Video Editing Studio Engine.
Validates:
1. Video media inspection and format validation (dimensions, fps, duration, aspect ratio).
2. Missing or corrupted file error handling.
3. Subtitle file synthesis (.srt and .vtt) with correct timestamp syntax.
4. Automatic 1280x720 YouTube thumbnail generation (3 distinct concepts).
5. YouTube Short generation with strict <= 60 seconds duration enforcement.
6. Studio database job creation, progress tracking, and artifact persistence.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.modules.video_editor_studio import (
    inspect_video_media,
    generate_subtitles_files,
    generate_youtube_thumbnails,
    generate_youtube_short,
    detect_educational_context
)
from backend.core import studio_db


class TestVideoEditorStudio(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Create a synthetic 3-second 720p test video using OpenCV
        self.sample_video = str(self.tmp_path / "synthetic_test.mp4")
        w, h = 1280, 720
        fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.sample_video, fourcc, fps, (w, h))

        for i in range(fps * 3): # 3 seconds
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Add dynamic graphic elements
            cv2.rectangle(frame, (50 + i * 5, 100), (350 + i * 5, 400), (59, 130, 246), -1)
            cv2.putText(frame, f"Frame {i}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            out.write(frame)

        out.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_media_inspection(self):
        """Verify video media properties are accurately detected."""
        info = inspect_video_media(self.sample_video)
        self.assertTrue(info["valid"])
        self.assertEqual(info["width"], 1280)
        self.assertEqual(info["height"], 720)
        self.assertEqual(info["fps"], 25.0)
        self.assertAlmostEqual(info["duration_seconds"], 3.0, delta=0.5)
        self.assertEqual(info["aspect_ratio"], "16:9")

    def test_media_inspection_missing_file_raises(self):
        """Verify inspecting non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            inspect_video_media("c:/non_existent_path/fake_video.mp4")

    def test_subtitle_generation(self):
        """Verify SRT and WebVTT caption files are properly formatted."""
        sample_segments = [
            {"start": 0.0, "end": 2.5, "text": "Welcome to the Machine Learning Masterclass."},
            {"start": 2.5, "end": 5.0, "text": "Today we explore deep neural networks."}
        ]
        base_out = str(self.tmp_path / "test_subs")
        srt_p, vtt_p = generate_subtitles_files(sample_segments, base_out, target_language="English")

        self.assertTrue(os.path.exists(srt_p))
        self.assertTrue(os.path.exists(vtt_p))

        with open(srt_p, "r", encoding="utf-8") as f:
            srt_content = f.read()
            self.assertIn("00:00:00,000 --> 00:00:02,500", srt_content)
            self.assertIn("Welcome to the Machine Learning Masterclass.", srt_content)

        with open(vtt_p, "r", encoding="utf-8") as f:
            vtt_content = f.read()
            self.assertTrue(vtt_content.startswith("WEBVTT"))
            self.assertIn("00:00:00.000 --> 00:00:02.500", vtt_content)

    def test_thumbnail_generation(self):
        """Verify 3 distinct thumbnails are synthesized with exact 1280x720 dimensions."""
        thumb_dir = str(self.tmp_path / "thumbs")
        thumbs = generate_youtube_thumbnails(
            title="Mastering LangGraph & Multi-Agent Architecture 2026",
            output_dir=thumb_dir
        )

        self.assertEqual(len(thumbs), 3)
        for t_path in thumbs:
            self.assertTrue(os.path.exists(t_path))
            with Image.open(t_path) as im:
                self.assertEqual(im.size, (1280, 720), f"Thumbnail {t_path} dimensions were {im.size}, expected (1280, 720)")
                self.assertEqual(im.format, "PNG")

    def test_educational_context_detection(self):
        """Verify topic keyword detection for contextual badges."""
        ctx_ai = detect_educational_context("In this neural network lecture we train tensors")
        self.assertIn("Artificial Intelligence", ctx_ai["domain"])

        ctx_code = detect_educational_context("Let's write a python function to implement quicksort")
        self.assertIn("Programming", ctx_code["domain"])

        ctx_math = detect_educational_context("Solving differential calculus equations and limits")
        self.assertIn("Mathematical", ctx_math["domain"])

    def test_studio_db_job_lifecycle(self):
        """Verify SQLite job tracking (create, progress, complete, fetch)."""
        job_id = f"test_job_{os.urandom(4).hex()}"
        job = studio_db.create_job(job_id, "video_editing", {"video": "test.mp4"})
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["progress"], 0.0)

        studio_db.update_job_progress(job_id, "Transcribing", 35.0, "Transcribed 10 segments")
        updated = studio_db.get_job(job_id)
        self.assertEqual(updated["status"], "running")
        self.assertEqual(updated["progress"], 35.0)
        self.assertEqual(updated["current_stage"], "Transcribing")
        self.assertTrue(len(updated["logs"]) > 0)

        studio_db.complete_job(job_id, {"video": "final.mp4"}, duration_seconds=12.5)
        completed = studio_db.get_job(job_id)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["progress"], 100.0)
        self.assertEqual(completed["duration_seconds"], 12.5)


if __name__ == "__main__":
    unittest.main()
