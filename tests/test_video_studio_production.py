"""
Automated Unit Tests for Autonomous Video Studio & Educational Content Engine
"""

import unittest
import os
import sys
import tempfile
from PIL import Image
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.modules.video_editor_studio import (
    generate_vertical_video_covers,
    generate_handwritten_study_notes,
    generate_engagement_animation_assets,
    render_stage_status_card,
    detect_silence_segments
)


class TestVideoStudioProduction(unittest.TestCase):
    def test_vertical_cover_generation_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            covers = generate_vertical_video_covers(
                title="Introduction to Graph Neural Networks",
                creator="prof_ai",
                output_dir=tmpdir,
                aspect_ratio="9:16 (Vertical)"
            )
            self.assertEqual(len(covers), 3, "Should generate 3 distinct cover concepts")
            for c_path in covers:
                self.assertTrue(os.path.exists(c_path))
                with Image.open(c_path) as img:
                    self.assertEqual(img.size, (1080, 1920), f"Cover must be strictly 1080x1920, got {img.size}")

    def test_handwritten_study_notes_pdf_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = os.path.join(tmpdir, "study_notes.pdf")
            script = """
            Artificial Neural Networks are computational systems inspired by the human brain.
            Layers include the input layer, multiple hidden layers, and an output layer.
            The Loss function L(w) computes the discrepancy between predicted and target values.
            Gradient Descent formulation:
            W_new = W_old - alpha * grad(L)
            Backpropagation propagates error gradients backwards through the computational graph.
            """
            pdf_path, png_paths = generate_handwritten_study_notes(
                video_title="Neural Networks & Backpropagation Masterclass",
                transcript_text=script,
                output_pdf_path=out_pdf,
                creator="prof_ai"
            )
            self.assertTrue(os.path.exists(pdf_path), "Study notes PDF must be compiled")
            self.assertGreater(os.path.getsize(pdf_path), 1000, "PDF size must be non-trivial")
            self.assertGreaterEqual(len(png_paths), 1, "Must generate at least 1 page PNG")
            for p_path in png_paths:
                self.assertTrue(os.path.exists(p_path))
                with Image.open(p_path) as img:
                    self.assertEqual(img.size, (1240, 1754), f"A4 page image should be 1240x1754, got {img.size}")

    def test_engagement_animation_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assets = generate_engagement_animation_assets(
                output_dir=tmpdir,
                brand_name="learn_ai"
            )
            expected_keys = ["like", "comment", "share", "follow", "bell"]
            for k in expected_keys:
                self.assertIn(k, assets, f"Missing asset key: {k}")
                file_path = assets[k]
                self.assertTrue(os.path.exists(file_path), f"Asset file missing: {file_path}")
                with Image.open(file_path) as img:
                    self.assertEqual(img.mode, "RGBA", "Engagement overlay must be transparent RGBA")

    def test_16_stage_status_card_rendering(self):
        html_queued = render_stage_status_card(
            current_stage_idx=0,
            total_stages=16,
            stage_name="Ready for 16-Stage Autonomous Editing",
            pct=0.0,
            elapsed_sec=0.0,
            status_label="queued",
            detail_note="Ready to start."
        )
        self.assertIn("0%", html_queued)
        self.assertIn("Ready for 16-Stage Autonomous Editing", html_queued)

        html_running = render_stage_status_card(
            current_stage_idx=8,
            total_stages=16,
            stage_name="Generating Multilingual Captions",
            pct=50.0,
            elapsed_sec=42.0,
            status_label="running",
            detail_note="Processing Kannada and Hindi captions."
        )
        self.assertIn("50%", html_running)
        self.assertIn("Stage 8 of 16", html_running)
        self.assertIn("00:42", html_running)

        html_complete = render_stage_status_card(
            current_stage_idx=16,
            total_stages=16,
            stage_name="Final Media Validation & Packaging",
            pct=100.0,
            elapsed_sec=95.0,
            status_label="completed",
            detail_note="All 16 stages validated successfully."
        )
        self.assertIn("100%", html_complete)
        self.assertIn("COMPLETED", html_complete.upper())


if __name__ == "__main__":
    unittest.main()
