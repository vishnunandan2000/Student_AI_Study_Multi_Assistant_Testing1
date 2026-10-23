"""
Unit and Integration Tests for Text-to-Video Generation Engine.
Validates:
1. Prompt cleaning and conversational trigger removal.
2. Domain styling detection for distinct topics (Space, Biology, Tech, Physics, etc.).
3. Multi-scene storyboard generation with topic-relevant scene breakdown.
4. Scene visual frame rendering and dimensions.
5. Subtitle SRT export and synchronization.
6. Complete end-to-end Text-to-Video generation and output MP4 validity.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from PIL import Image

# Ensure project root is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.modules.video_engine import (
    StoryboardEngine,
    detect_domain_styling,
    render_scene_card_image,
    export_subtitles,
    generate_text_to_video
)


class TestTextToVideoEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prompt_cleaning(self):
        """Verify conversational triggers are removed to isolate the core topic."""
        prompts = [
            ("generate a video about the solar system", "the solar system"),
            ("create a video of photosynthesis in plants", "photosynthesis in plants"),
            ("explain how quantum computers work", "quantum computers work"),
            ("show me a video about ancient Egypt", "ancient Egypt"),
            ("make a video explaining neural networks", "explaining neural networks")
        ]
        for raw, expected in prompts:
            cleaned = StoryboardEngine.clean_prompt(raw)
            self.assertEqual(cleaned.lower(), expected.lower())

    def test_domain_styling_detection(self):
        """Verify domain styles (colors, icons, category) match prompt subjects."""
        # Astronomy
        space_style = detect_domain_styling("Planetary orbits in the Solar System")
        self.assertIn("Space", space_style["category"])
        self.assertEqual(space_style["icon"], "🪐")

        # Biology
        bio_style = detect_domain_styling("Photosynthesis in green leaves and plants")
        self.assertIn("Biology", bio_style["category"])
        self.assertEqual(bio_style["icon"], "🌿")

        # Tech / AI
        tech_style = detect_domain_styling("Deep neural networks and artificial intelligence algorithms")
        self.assertIn("Technology", tech_style["category"])
        self.assertEqual(tech_style["icon"], "💻")

        # Physics
        physics_style = detect_domain_styling("Quantum mechanics and atomic structure")
        self.assertIn("Physics", physics_style["category"])
        self.assertEqual(physics_style["icon"], "🔬")

    def test_storyboard_generation_relevance(self):
        """Verify storyboard generator produces 3 cohesive, relevant scenes."""
        prompt = "Photosynthesis in green plants"
        scenes = StoryboardEngine.generate_storyboard_from_prompt(prompt, num_scenes=3)

        self.assertEqual(len(scenes), 3)
        for sc in scenes:
            self.assertIn("scene_num", sc)
            self.assertIn("title", sc)
            self.assertIn("visual_prompt", sc)
            self.assertIn("narration", sc)
            self.assertIn("key_point", sc)
            self.assertTrue(len(sc["title"]) > 0)
            self.assertTrue(len(sc["visual_prompt"]) > 0)
            self.assertTrue(len(sc["narration"]) > 0)

        # Check content relevance: scenes must relate to plant/photosynthesis/energy
        all_text = " ".join(
            (sc["title"] + " " + sc["visual_prompt"] + " " + sc["narration"]).lower()
            for sc in scenes
        )
        self.assertTrue(
            any(k in all_text for k in ["plant", "photo", "leaf", "chlorophyll", "light", "sun", "energy", "glucose", "cell"])
        )

    def test_scene_frame_rendering(self):
        """Verify scene card image is rendered with correct resolution and valid pixels."""
        scene = {
            "scene_num": 1,
            "title": "Sunlight Absorption",
            "visual_prompt": "close-up of plant leaf cells absorbing sunlight rays, photorealistic 8k",
            "narration": "Chlorophyll pigments absorb sunlight to begin photosynthesis.",
            "key_point": "Light absorption"
        }
        img = render_scene_card_image(scene, "Photosynthesis", width=1280, height=720, temp_dir=self.tmp_path)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (1280, 720))
        self.assertEqual(img.mode, "RGB")

    def test_subtitle_export(self):
        """Verify export_subtitles produces a formatted SRT file matching the scenes."""
        scenes = [
            {"scene_num": 1, "title": "Intro", "narration": "Welcome to the solar system."},
            {"scene_num": 2, "title": "Planets", "narration": "Eight planets orbit the sun."}
        ]
        srt_file = self.tmp_path / "test_sub.srt"
        export_subtitles(scenes, [4.5, 5.0], str(srt_file))

        self.assertTrue(srt_file.exists())
        with open(srt_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("00:00:00,000 --> 00:00:04,500", content)
        self.assertIn("Welcome to the solar system.", content)
        self.assertIn("Eight planets orbit the sun.", content)

    def test_end_to_end_text_to_video(self):
        """Verify end-to-end Text-to-Video generation produces a valid, playable MP4 file."""
        prompt = "How rockets launch into space"
        out_mp4 = self.tmp_path / "test_rocket_launch.mp4"

        final_path = generate_text_to_video(
            prompt=prompt,
            output_path=str(out_mp4),
            resolution="720p",
            num_scenes=2  # 2 scenes for fast test execution
        )

        self.assertTrue(os.path.exists(final_path))
        self.assertGreater(os.path.getsize(final_path), 50000)  # > 50KB

        # Verify video can be read and has valid duration and audio
        from moviepy import VideoFileClip
        clip = VideoFileClip(final_path)
        try:
            self.assertGreater(clip.duration, 5.0)
            self.assertEqual(clip.size, [1280, 720])
            self.assertIsNotNone(clip.audio)
        finally:
            clip.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
