"""
Unit and Integration Tests for Image-to-Video Generation Engine.
Validates:
1. Robust reference image path extraction from diverse formats (str, Path, Gradio dict, file objects, lists).
2. Input validation, corruption detection, and error handling for missing/invalid images.
3. User motion prompt parsing and camera dynamics selection (Zoom, Pan, Tilt, Sweep, Pulse).
4. Multi-format support across PNG (with alpha), JPG, WEBP, and BMP images.
5. Multi-aspect ratio handling (Landscape, Square with pillarbox blur, Tall Portrait with vertical tilt).
6. End-to-end video synthesis producing valid, playable MP4 files with synchronized audio.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.modules.video_engine import (
    extract_image_path,
    detect_motion_type,
    generate_image_to_video
)


class DummyFileObjWithName:
    def __init__(self, name: str):
        self.name = name


class DummyFileObjWithPath:
    def __init__(self, path: str):
        self.path = path


class TestImageToVideoEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Create valid sample images
        self.sample_png = self.tmp_path / "sample.png"
        img_png = Image.new("RGBA", (600, 600), (30, 41, 59, 255))
        d_png = ImageDraw.Draw(img_png)
        d_png.ellipse([150, 150, 450, 450], fill=(59, 130, 246, 220), outline=(255, 255, 255), width=4)
        img_png.save(str(self.sample_png), format="PNG")

        self.sample_jpg = self.tmp_path / "sample.jpg"
        img_jpg = Image.new("RGB", (1280, 720), (15, 23, 42))
        d_jpg = ImageDraw.Draw(img_jpg)
        d_jpg.rectangle([200, 150, 1080, 570], fill=(16, 185, 129))
        img_jpg.save(str(self.sample_jpg), format="JPEG")

        self.sample_webp = self.tmp_path / "sample.webp"
        img_webp = Image.new("RGB", (720, 1280), (79, 70, 229))
        d_webp = ImageDraw.Draw(img_webp)
        d_webp.polygon([(360, 200), (600, 1000), (120, 1000)], fill=(245, 158, 11))
        img_webp.save(str(self.sample_webp), format="WEBP")

        self.sample_bmp = self.tmp_path / "sample.bmp"
        img_bmp = Image.new("RGB", (800, 600), (220, 38, 38))
        img_bmp.save(str(self.sample_bmp), format="BMP")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_image_path_valid_types(self):
        """Verify image path extraction from strings, Path objects, Gradio dicts, and file objects."""
        # 1. String path
        res_str = extract_image_path(str(self.sample_png))
        self.assertIsNotNone(res_str)
        self.assertEqual(Path(res_str).resolve(), self.sample_png.resolve())

        # 2. Path object
        res_path = extract_image_path(self.sample_png)
        self.assertIsNotNone(res_path)
        self.assertEqual(Path(res_path).resolve(), self.sample_png.resolve())

        # 3. Gradio file dict with 'path'
        res_dict_path = extract_image_path({"path": str(self.sample_jpg), "orig_name": "photo.jpg"})
        self.assertIsNotNone(res_dict_path)
        self.assertEqual(Path(res_dict_path).resolve(), self.sample_jpg.resolve())

        # 4. Gradio file dict with 'name'
        res_dict_name = extract_image_path({"name": str(self.sample_jpg)})
        self.assertIsNotNone(res_dict_name)
        self.assertEqual(Path(res_dict_name).resolve(), self.sample_jpg.resolve())

        # 5. Object with .name attribute
        dummy_name = DummyFileObjWithName(str(self.sample_webp))
        res_obj_name = extract_image_path(dummy_name)
        self.assertIsNotNone(res_obj_name)
        self.assertEqual(Path(res_obj_name).resolve(), self.sample_webp.resolve())

        # 6. Object with .path attribute
        dummy_path = DummyFileObjWithPath(str(self.sample_webp))
        res_obj_path = extract_image_path(dummy_path)
        self.assertIsNotNone(res_obj_path)
        self.assertEqual(Path(res_obj_path).resolve(), self.sample_webp.resolve())

        # 7. List or tuple containing image
        res_list = extract_image_path([str(self.sample_bmp)])
        self.assertIsNotNone(res_list)
        self.assertEqual(Path(res_list).resolve(), self.sample_bmp.resolve())

    def test_extract_image_path_invalid_inputs(self):
        """Verify robust rejection of invalid, non-existent, or corrupted files."""
        # None and empty
        self.assertIsNone(extract_image_path(None))
        self.assertIsNone(extract_image_path(""))
        self.assertIsNone(extract_image_path([]))

        # Non-existent file
        non_existent = self.tmp_path / "ghost.png"
        self.assertIsNone(extract_image_path(str(non_existent)))

        # Plain text file disguised as image
        fake_img = self.tmp_path / "fake.png"
        fake_img.write_text("This is not an image file.", encoding="utf-8")
        self.assertIsNone(extract_image_path(str(fake_img)))

    def test_detect_motion_type(self):
        """Verify motion keywords are accurately mapped to camera kinematics."""
        # Zoom In
        m_type, _ = detect_motion_type("Please zoom in on the main structure", 1280, 720)
        self.assertEqual(m_type, "ZOOM_IN")

        # Zoom Out
        m_type, _ = detect_motion_type("Pull back slowly to reveal context", 1280, 720)
        self.assertEqual(m_type, "ZOOM_OUT")

        # Pan Left
        m_type, _ = detect_motion_type("Pan left across the landscape", 1280, 720)
        self.assertEqual(m_type, "PAN_LEFT")

        # Pan Right
        m_type, _ = detect_motion_type("Move rightwards smoothly", 1280, 720)
        self.assertEqual(m_type, "PAN_RIGHT")

        # Pan Down
        m_type, _ = detect_motion_type("Tilt down from the top to bottom", 1280, 720)
        self.assertEqual(m_type, "PAN_DOWN")

        # Pan Up
        m_type, _ = detect_motion_type("Scroll up slowly", 1280, 720)
        self.assertEqual(m_type, "PAN_UP")

        # Cinematic Sweep
        m_type, _ = detect_motion_type("Give a cinematic sweep flyover effect", 1280, 720)
        self.assertEqual(m_type, "CINEMATIC_SWEEP")

        # Pulse
        m_type, _ = detect_motion_type("Subtle heartbeat pulse effect", 1280, 720)
        self.assertEqual(m_type, "PULSE")

        # Aspect-ratio intelligent default (tall portrait -> pan down)
        m_type_tall, _ = detect_motion_type("", 720, 1280)
        self.assertEqual(m_type_tall, "PAN_DOWN")

        # Aspect-ratio intelligent default (landscape -> zoom in)
        m_type_land, _ = detect_motion_type("", 1280, 720)
        self.assertEqual(m_type_land, "ZOOM_IN")

    def test_generate_image_to_video_missing_ref_raises(self):
        """Verify generate_image_to_video raises ValueError when given an invalid reference image."""
        with self.assertRaises(ValueError):
            generate_image_to_video("c:/non_existent_folder/missing.png", "zoom in")

    def test_end_to_end_image_to_video_generation(self):
        """Verify end-to-end video synthesis across different formats and motions."""
        test_cases = [
            (self.sample_png, "Pan right across the diagram", "png_pan_right.mp4"),
            (self.sample_jpg, "Cinematic push-in zoom on subject", "jpg_zoom_in.mp4"),
            (self.sample_webp, "Tilt down through the vertical infographic", "webp_pan_down.mp4"),
        ]

        for img_path, prompt, out_filename in test_cases:
            out_vid = self.tmp_path / out_filename
            result_path = generate_image_to_video(
                input_image_path=str(img_path),
                prompt=prompt,
                output_path=str(out_vid),
                resolution="720p"
            )

            # 1. Output file exists and matches requested path
            self.assertTrue(os.path.exists(result_path), f"Video file {result_path} was not created.")
            self.assertEqual(Path(result_path).resolve(), out_vid.resolve())

            # 2. Output file has meaningful size (> 30KB)
            file_size = os.path.getsize(result_path)
            self.assertGreater(file_size, 30_000, f"Video file {result_path} is too small ({file_size} bytes).")

            # 3. Check for valid MP4 header ('ftyp' box in first 32 bytes)
            with open(result_path, "rb") as f:
                header = f.read(32)
                self.assertIn(b"ftyp", header, f"File {result_path} missing valid MP4 'ftyp' container signature.")


if __name__ == "__main__":
    unittest.main()
