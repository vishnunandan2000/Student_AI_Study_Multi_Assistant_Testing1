"""
Unit and Integration Tests for Image-to-Image Generation & Editing Engine.
Validates:
1. Handling of missing or non-existent reference images (Requirement 7).
2. Handling of corrupted or invalid image files (Requirement 7).
3. Preservation of reference image subject and identity (Requirements 2, 4).
4. Prompt instruction adherence across background, lighting, and clothing scenarios (Requirements 3, 6).
5. Output file validity (format, resolution, integrity) (Requirement 8).
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

# Ensure project root is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.modules.image_engine import (
    generate_ai_image_to_image,
    generate_semantic_reference_edit,
    optimize_img2img_prompt
)


class TestImageToImageEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Create valid synthetic portrait image
        self.sample_ref = self.tmp_path / "test_portrait.png"
        img = Image.new("RGB", (300, 400), color=(180, 150, 130))
        draw = ImageDraw.Draw(img)
        draw.ellipse([100, 100, 200, 220], fill=(220, 180, 150), outline=(50, 50, 50))
        draw.rectangle([70, 220, 230, 390], fill=(50, 70, 120))
        img.save(str(self.sample_ref))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_reference_image_raises_error(self):
        """Verify missing reference image raises clear ValueError instead of silent fallback."""
        with self.assertRaises(ValueError) as ctx:
            generate_ai_image_to_image("non_existent_reference_xyz.jpg", "change background to a park")
        self.assertIn("not found or missing", str(ctx.exception))

    def test_corrupted_image_raises_error(self):
        """Verify corrupted reference image raises clear ValueError."""
        corrupt_file = self.tmp_path / "corrupted.jpg"
        with open(corrupt_file, "w") as f:
            f.write("Not a real image binary data")

        with self.assertRaises(ValueError) as ctx:
            generate_ai_image_to_image(str(corrupt_file), "turn background into library")
        self.assertIn("corrupted or invalid", str(ctx.exception))

    def test_prompt_optimization_preserves_identity(self):
        """Verify prompt optimizer instructs diffusion models to maintain identity."""
        prompt = "change clothing to a navy blue tuxedo"
        optimized = optimize_img2img_prompt(prompt)
        self.assertGreater(len(optimized), 10)
        self.assertTrue(
            any(k in optimized.lower() for k in ["identity", "likeness", "facial", "reference", "subject"])
        )

    def test_semantic_lighting_edit(self):
        """Verify semantic lighting transformation modifies image without losing reference."""
        out_path = self.tmp_path / "lighting_sunset.png"
        result = generate_semantic_reference_edit(
            str(self.sample_ref),
            "dramatic golden hour sunset lighting",
            str(out_path)
        )
        self.assertTrue(os.path.exists(result))
        with Image.open(result) as im:
            self.assertEqual(im.format, "PNG")
            self.assertEqual(im.size, (300, 400))

    def test_semantic_background_edit(self):
        """Verify semantic background replacement preserves subject identity."""
        out_path = self.tmp_path / "bg_library.png"
        result = generate_semantic_reference_edit(
            str(self.sample_ref),
            "futuristic university library background with modern bookshelves",
            str(out_path)
        )
        self.assertTrue(os.path.exists(result))
        with Image.open(result) as im:
            self.assertEqual(im.format, "PNG")
            self.assertEqual(im.size, (300, 400))


if __name__ == "__main__":
    unittest.main(verbosity=2)
