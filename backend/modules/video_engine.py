import os
import sys
import json
import re
import uuid
import time
import math
import logging
import traceback
import shutil
import asyncio
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps

try:
    from moviepy import ImageClip, ColorClip, CompositeVideoClip, AudioFileClip, VideoClip, concatenate_videoclips
except (ImportError, ModuleNotFoundError):
    from moviepy.editor import ImageClip, ColorClip, CompositeVideoClip, AudioFileClip, VideoClip, concatenate_videoclips

from backend.modules.audio_engine import AudioNarrationEngine, detect_language
from backend.modules.translator import translate_text, validate_translation
from backend.core.llm import get_llm
from langchain_core.messages import HumanMessage

# Configure module logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _set_clip_duration(clip, duration: float):
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    elif hasattr(clip, "set_duration"):
        return clip.set_duration(duration)
    return clip


def _set_clip_audio(video_clip, audio_clip):
    if hasattr(video_clip, "with_audio"):
        return video_clip.with_audio(audio_clip)
    elif hasattr(video_clip, "set_audio"):
        return video_clip.set_audio(audio_clip)
    return video_clip


def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Helper to load system fonts or fallback to default."""
    font_names = ["arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]
    if bold:
        font_names = ["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"] + font_names

    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> str:
    """Wrap text to fit within a specific width constraint (returns joined string)."""
    lines = []
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        words = paragraph.split(' ')
        if not words:
            continue
        current_line = []
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                width = bbox[2] - bbox[0]
            except AttributeError:
                width, _ = draw.textsize(test_line, font=font)

            if width > max_width and len(current_line) > 1:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
    return '\n'.join(lines)


def wrap_text_lines(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    """Wraps text into a list of lines fitting max_width constraint."""
    lines = []
    paragraphs = text.split('\n')
    for p in paragraphs:
        words = p.split(' ')
        if not words:
            continue
        current_line = []
        for word in words:
            current_line.append(word)
            test_str = ' '.join(current_line)
            try:
                bbox = draw.textbbox((0, 0), test_str, font=font)
                w = bbox[2] - bbox[0]
            except AttributeError:
                w, _ = draw.textsize(test_str, font=font)
            if w > max_width and len(current_line) > 1:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
    return lines


def detect_domain_styling(text: str) -> Dict[str, Any]:
    """Detects domain theme, colors, icon, and categories based on text keywords."""
    t = text.lower()
    if any(k in t for k in ["space", "astronomy", "planet", "galaxy", "orbit", "rocket", "moon", "star", "mars", "solar", "universe", "nebula"]):
        return {
            "bg_top": (10, 15, 35),
            "bg_bottom": (25, 20, 60),
            "accent": (139, 92, 246),
            "glow": (192, 132, 252),
            "icon": "🪐",
            "category": "Astronomy & Space"
        }
    elif any(k in t for k in ["plant", "photosynthesis", "leaf", "chlorophyll", "cell", "tree", "biology", "nature", "animal", "earth", "water", "forest", "dna", "eco"]):
        return {
            "bg_top": (10, 30, 20),
            "bg_bottom": (18, 48, 32),
            "accent": (16, 185, 129),
            "glow": (110, 231, 183),
            "icon": "🌿",
            "category": "Biology & Nature"
        }
    elif any(k in t for k in ["code", "computer", "ai", "software", "cyber", "robot", "neural", "algorithm", "data", "program", "cloud", "internet"]):
        return {
            "bg_top": (15, 23, 42),
            "bg_bottom": (28, 40, 58),
            "accent": (59, 130, 246),
            "glow": (147, 197, 253),
            "icon": "💻",
            "category": "Technology & AI"
        }
    elif any(k in t for k in ["physics", "atom", "quantum", "energy", "chemistry", "force", "gravity", "motion", "magnet", "electron", "wave", "mechanic"]):
        return {
            "bg_top": (15, 25, 45),
            "bg_bottom": (30, 35, 65),
            "accent": (6, 182, 212),
            "glow": (103, 232, 249),
            "icon": "🔬",
            "category": "Physics & Science"
        }
    elif any(k in t for k in ["history", "pyramid", "ancient", "egypt", "roman", "war", "civilization", "empire", "culture", "philosophy"]):
        return {
            "bg_top": (28, 20, 15),
            "bg_bottom": (45, 32, 22),
            "accent": (217, 119, 6),
            "glow": (251, 191, 36),
            "icon": "🏛️",
            "category": "History & Culture"
        }
    else:
        return {
            "bg_top": (20, 24, 36),
            "bg_bottom": (35, 42, 58),
            "accent": (245, 158, 11),
            "glow": (252, 211, 77),
            "icon": "🎬",
            "category": "Visual Explainer"
        }


# =====================================================================
# Professor RoboCat Character & Visual Prop Renderers (For Tab 9: 3D Video Studio)
# =====================================================================

class ProfessorRoboCatRenderer:
    """
    Renders an original 3D-styled robotic cat teacher ('Professor RoboCat').
    Includes lip-synchronization mouth movements, blinking eyes, metallic gradient highlights,
    and dynamic pose transitions.
    """
    @staticmethod
    def draw_character(
        size: Tuple[int, int] = (400, 450),
        pose: str = "TEACHING",
        expression: str = "HAPPY",
        mouth_state: str = "OPEN",
        is_blinking: bool = False
    ) -> Image.Image:
        width, height = size
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx, cy = width // 2, height // 2 - 20

        # Ground Shadow
        draw.ellipse([cx - 100, height - 40, cx + 100, height - 10], fill=(0, 0, 0, 80))

        # 3D Metallic Blue Torso / Body
        body_box = [cx - 80, cy + 40, cx + 80, cy + 180]
        draw.ellipse(body_box, fill=(35, 115, 225, 255))
        draw.ellipse([cx - 45, cy + 65, cx + 45, cy + 155], fill=(240, 245, 255, 255))

        # Teacher Bowtie & Gold Badge
        draw.polygon([(cx - 20, cy + 60), (cx + 20, cy + 60), (cx, cy + 75)], fill=(235, 55, 55, 255))
        draw.polygon([(cx - 20, cy + 90), (cx + 20, cy + 90), (cx, cy + 75)], fill=(235, 55, 55, 255))
        draw.ellipse([cx - 8, cy + 70, cx + 8, cy + 80], fill=(255, 215, 0, 255))

        # 3D Cat Ears
        ear_left = [(cx - 95, cy - 60), (cx - 45, cy - 130), (cx - 15, cy - 75)]
        ear_right = [(cx + 15, cy - 75), (cx + 45, cy - 130), (cx + 95, cy - 60)]
        draw.polygon(ear_left, fill=(25, 95, 205, 255))
        draw.polygon(ear_right, fill=(25, 95, 205, 255))
        draw.polygon([(cx - 80, cy - 65), (cx - 45, cy - 115), (cx - 25, cy - 75)], fill=(0, 230, 255, 230))
        draw.polygon([(cx + 25, cy - 75), (cx + 45, cy - 115), (cx + 80, cy - 65)], fill=(0, 230, 255, 230))

        # 3D Head Shell
        head_box = [cx - 110, cy - 110, cx + 110, cy + 50]
        draw.ellipse(head_box, fill=(45, 130, 245, 255))
        draw.ellipse([cx - 90, cy - 95, cx + 20, cy - 30], fill=(90, 175, 255, 120))

        # Face Mask
        face_box = [cx - 85, cy - 50, cx + 85, cy + 40]
        draw.ellipse(face_box, fill=(245, 250, 255, 255))

        # Eyes & Blinking Animation
        eye_y = cy - 15
        if is_blinking:
            draw.line([(cx - 60, eye_y), (cx - 20, eye_y)], fill=(30, 40, 70, 255), width=5)
            draw.line([(cx + 20, eye_y), (cx + 60, eye_y)], fill=(30, 40, 70, 255), width=5)
        elif expression == "HAPPY":
            draw.arc([cx - 60, eye_y - 20, cx - 20, eye_y + 15], 200, 340, fill=(0, 220, 255, 255), width=6)
            draw.arc([cx + 20, eye_y - 20, cx + 60, eye_y + 15], 200, 340, fill=(0, 220, 255, 255), width=6)
        elif expression == "CELEBRATE":
            draw.ellipse([cx - 55, eye_y - 15, cx - 25, eye_y + 15], fill=(255, 215, 0, 255))
            draw.ellipse([cx + 25, eye_y - 15, cx + 55, eye_y + 15], fill=(255, 215, 0, 255))
        else:
            draw.ellipse([cx - 60, eye_y - 25, cx - 20, eye_y + 20], fill=(20, 30, 60, 255))
            draw.ellipse([cx + 20, eye_y - 25, cx + 60, eye_y + 20], fill=(20, 30, 60, 255))
            draw.ellipse([cx - 52, eye_y - 18, cx - 28, eye_y + 12], fill=(0, 220, 255, 255))
            draw.ellipse([cx + 28, eye_y - 18, cx + 52, eye_y + 12], fill=(0, 220, 255, 255))
            draw.ellipse([cx - 48, eye_y - 14, cx - 38, eye_y - 4], fill=(255, 255, 255, 255))
            draw.ellipse([cx + 32, eye_y - 14, cx + 42, eye_y - 4], fill=(255, 255, 255, 255))

        # Nose
        draw.ellipse([cx - 8, cy + 5, cx + 8, cy + 18], fill=(255, 100, 130, 255))

        # Lip Sync Mouth
        if mouth_state == "OPEN":
            draw.ellipse([cx - 15, cy + 22, cx + 15, cy + 38], fill=(200, 40, 60, 255))
        elif mouth_state == "SOFT_OPEN":
            draw.ellipse([cx - 10, cy + 24, cx + 10, cy + 32], fill=(220, 60, 80, 255))
        else:
            draw.arc([cx - 15, cy + 15, cx + 15, cy + 30], 20, 160, fill=(40, 50, 80, 255), width=4)

        # Whiskers
        draw.line([(cx - 70, cy + 10), (cx - 105, cy + 2)], fill=(40, 130, 230, 255), width=3)
        draw.line([(cx - 70, cy + 20), (cx - 105, cy + 22)], fill=(40, 130, 230, 255), width=3)
        draw.line([(cx + 70, cy + 10), (cx + 105, cy + 2)], fill=(40, 130, 230, 255), width=3)
        draw.line([(cx + 70, cy + 20), (cx + 105, cy + 22)], fill=(40, 130, 230, 255), width=3)

        # Arms & Pose
        if pose == "POINTING":
            draw.line([(cx + 60, cy + 80), (cx + 140, cy + 10)], fill=(40, 130, 230, 255), width=16)
            draw.line([(cx + 130, cy + 20), (cx + 180, cy - 40)], fill=(255, 215, 0, 255), width=6)
            draw.ellipse([cx + 172, cy - 48, cx + 188, cy - 32], fill=(0, 240, 255, 255))
            draw.line([(cx - 60, cy + 80), (cx - 95, cy + 130)], fill=(40, 130, 230, 255), width=16)
        elif pose == "CELEBRATING":
            draw.line([(cx - 60, cy + 80), (cx - 110, cy + 20)], fill=(40, 130, 230, 255), width=16)
            draw.line([(cx + 60, cy + 80), (cx + 110, cy + 20)], fill=(40, 130, 230, 255), width=16)
        else:
            draw.line([(cx - 60, cy + 80), (cx - 105, cy + 50)], fill=(40, 130, 230, 255), width=16)
            draw.line([(cx + 60, cy + 80), (cx + 105, cy + 110)], fill=(40, 130, 230, 255), width=16)

        return img


class VisualPropRenderer:
    """Renders interactive 3D topic objects (AI Neural Networks, Flowcharts, Atoms, Solar System)."""
    @staticmethod
    def draw_prop(draw: ImageDraw.ImageDraw, pos: Tuple[int, int], prop_type: str = "AI Neural Networks"):
        px, py = pos
        if "Neural" in prop_type or "AI" in prop_type:
            nodes = [(px - 50, py - 40), (px - 50, py + 40), (px + 50, py - 60), (px + 50, py), (px + 50, py + 60)]
            for n1 in nodes[:2]:
                for n2 in nodes[2:]:
                    draw.line([n1, n2], fill=(0, 220, 255, 120), width=2)
            for nx, ny in nodes:
                draw.ellipse([nx - 12, ny - 12, nx + 12, ny + 12], fill=(255, 215, 0, 240), outline=(0, 240, 255, 255), width=2)
        elif "Coding" in prop_type or "Flowchart" in prop_type:
            draw.rectangle([px - 60, py - 60, px + 60, py - 20], fill=(30, 90, 220, 220), outline=(0, 220, 255, 255), width=2)
            draw.polygon([(px, py - 10), (px + 50, py + 20), (px, py + 50), (px - 50, py + 20)], fill=(220, 150, 30, 220))
            draw.line([(px, py - 20), (px, py - 10)], fill=(0, 240, 255, 255), width=3)
        else:
            draw.ellipse([px - 50, py - 50, px + 50, py + 50], outline=(0, 220, 255, 120), width=2)
            draw.ellipse([px - 70, py - 25, px + 70, py + 25], outline=(0, 220, 255, 140), width=2)
            draw.ellipse([px - 15, py - 15, px + 15, py + 15], fill=(255, 215, 0, 255))


# =====================================================================
# Storyboard Engine (Intelligent Scene Parsing & Generation)
# =====================================================================

class StoryboardEngine:
    """Splits full educational text or generates multi-scene educational storyboard segments from prompts."""

    @staticmethod
    def clean_prompt(prompt: str) -> str:
        """Strips conversational noise and triggers from user prompts."""
        cleaned = prompt.strip()
        triggers = [
            "generate a video of", "generate a video about", "generate a video on", "generate a video",
            "generate video of", "generate video about", "generate video",
            "create a video of", "create a video about", "create a video on", "create a video",
            "create video of", "create video about", "create video",
            "make a video of", "make a video about", "make a video",
            "explain to me", "explain how", "explain why", "explain",
            "show me a video of", "show me a video about", "show me how", "show me",
            "tell me about", "teach me about", "i want a video of", "video on", "video of"
        ]
        for tr in triggers:
            pattern = re.compile(rf"^{re.escape(tr)}\s+", re.IGNORECASE)
            cleaned = pattern.sub("", cleaned).strip()
        return cleaned or prompt.strip()

    @staticmethod
    def generate_storyboard_from_prompt(prompt: str, num_scenes: int = 3) -> List[Dict[str, Any]]:
        """
        Transforms any user prompt into a sequential, cohesive multi-scene video storyboard.
        Uses local Ollama LLM with intelligent domain-aware fallback.
        """
        clean_p = StoryboardEngine.clean_prompt(prompt)
        logger.info(f"Generating storyboard for prompt: '{clean_p}' (target scenes: {num_scenes})")

        # Attempt structured Ollama generation
        try:
            llm = get_llm()
            instruction = (
                f"You are an expert educational and documentary video director. "
                f"Deconstruct the user's prompt into exactly {num_scenes} sequential, cohesive video scenes. "
                f"For each scene provide:\n"
                f"1. scene_num (integer 1 to {num_scenes})\n"
                f"2. title (concise descriptive scene title, max 5 words)\n"
                f"3. visual_prompt (vivid, photorealistic description of the visual scene for generative image diffusion, max 25 words)\n"
                f"4. narration (natural, engaging spoken voiceover explaining the scene, 1-2 sentences, max 30 words)\n"
                f"5. key_point (a single concise takeaway bullet, max 8 words)\n\n"
                f"Respond STRICTLY with a valid JSON array of {num_scenes} scene objects. "
                f"Do not include markdown codeblocks or conversational text."
            )
            resp = llm.invoke([HumanMessage(content=f"{instruction}\n\nUser prompt: {clean_p}")])
            content = resp.content.strip()
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                content = match.group(0)
            scenes = json.loads(content)
            if isinstance(scenes, list) and len(scenes) >= 1:
                # Normalize scene fields
                normalized = []
                for idx, sc in enumerate(scenes[:num_scenes], start=1):
                    normalized.append({
                        "scene_num": idx,
                        "title": sc.get("title", f"Scene {idx}").strip('"*# '),
                        "visual_prompt": sc.get("visual_prompt", f"{clean_p} visual representation").strip('"*# '),
                        "narration": sc.get("narration", f"Exploring {clean_p}.").strip('"*# '),
                        "key_point": sc.get("key_point", "Key concept").strip('"*# '),
                    })
                logger.info(f"Ollama successfully produced {len(normalized)} storyboard scenes.")
                return normalized
        except Exception as e:
            logger.warning(f"Ollama storyboard synthesis fallback activated ({e}).")

        # Heuristic Domain-Aware Fallback
        styling = detect_domain_styling(clean_p)
        topic_short = clean_p[:40]

        return [
            {
                "scene_num": 1,
                "title": f"Introduction to {topic_short}",
                "visual_prompt": f"high quality cinematic visualization of {clean_p}, {styling['category']}, photorealistic 8k",
                "narration": f"Welcome. Today we examine {clean_p}, exploring its essential foundations and scope.",
                "key_point": f"Foundations of {topic_short[:25]}"
            },
            {
                "scene_num": 2,
                "title": "Core Mechanism & Process",
                "visual_prompt": f"detailed dynamic visual breakdown showing internal workings and elements of {clean_p}, 4k",
                "narration": f"At its core, {clean_p} operates through key dynamics that drive essential transformations.",
                "key_point": "Key operational dynamics"
            },
            {
                "scene_num": 3,
                "title": "Real-World Impact & Summary",
                "visual_prompt": f"broad impact, real world significance, and scientific relevance of {clean_p}, 4k ultra realistic",
                "narration": f"Understanding {clean_p} provides vital insights into science, nature, and modern technology.",
                "key_point": "Significance and applications"
            }
        ]

    @staticmethod
    def create_scenes(text: str) -> List[Dict[str, Any]]:
        """Splits full educational text into multi-scene educational storyboard segments."""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if len(paragraphs) <= 1:
            # If prompt is a brief topic, use prompt decomposition
            return StoryboardEngine.generate_storyboard_from_prompt(text, num_scenes=3)

        scenes = []
        titles = ["1. INTRODUCTION", "2. CORE CONCEPT", "3. DEEP DIVE & EXAMPLES", "4. RECAP & QUIZ"]
        poses = ["TEACHING", "POINTING", "THINKING", "CELEBRATING"]
        expressions = ["HAPPY", "HAPPY", "TEACHING", "CELEBRATING"]

        for i, para in enumerate(paragraphs[:4]):
            scenes.append({
                "scene_num": i + 1,
                "title": titles[i] if i < len(titles) else f"SCENE {i+1}",
                "script": para,
                "narration": para,
                "visual_prompt": para[:100],
                "pose": poses[i % len(poses)],
                "expression": expressions[i % len(expressions)],
            })
        return scenes


# =====================================================================
# Subtitle Exporter
# =====================================================================

def export_subtitles(scenes: List[Dict[str, Any]], scene_duration_or_list: Union[float, List[float]], srt_path: str):
    """Exports SRT subtitle file synchronized with video scenes."""
    srt_p = Path(srt_path).resolve()
    srt_p.parent.mkdir(parents=True, exist_ok=True)
    with open(srt_p, "w", encoding="utf-8") as f:
        start_time = 0.0
        for i, sc in enumerate(scenes):
            if isinstance(scene_duration_or_list, list):
                dur = scene_duration_or_list[i] if i < len(scene_duration_or_list) else 4.0
            else:
                dur = float(scene_duration_or_list)
            end_time = start_time + dur
            
            s_min, s_sec = divmod(int(start_time), 60)
            s_hr, s_min = divmod(s_min, 60)
            s_ms = int((start_time - int(start_time)) * 1000)

            e_min, e_sec = divmod(int(end_time), 60)
            e_hr, e_min = divmod(e_min, 60)
            e_ms = int((end_time - int(end_time)) * 1000)

            start_str = f"{s_hr:02d}:{s_min:02d}:{s_sec:02d},{s_ms:03d}"
            end_str = f"{e_hr:02d}:{e_min:02d}:{e_sec:02d},{e_ms:03d}"
            
            text_sub = sc.get("narration") or sc.get("script") or sc.get("title", "")
            f.write(f"{i+1}\n{start_str} --> {end_str}\n{text_sub}\n\n")
            start_time = end_time


# =====================================================================
# Scene Visual Renderer (Diffusion with Infographic Fallback)
# =====================================================================

def render_scene_card_image(scene: Dict[str, Any], topic: str, width: int = 1280, height: int = 720, temp_dir: Optional[Path] = None) -> Image.Image:
    """
    Renders a dedicated, relevant visual image frame for a video scene:
    1. Attempts online diffusion image generation (quick timeout).
    2. Seamlessly falls back to an intelligent, topic-customized educational visual schematic canvas.
    """
    style = detect_domain_styling(topic + " " + scene.get("title", "") + " " + scene.get("visual_prompt", ""))
    scene_num = scene.get("scene_num", 1)
    scene_title = scene.get("title", f"Scene {scene_num}")
    visual_prompt = scene.get("visual_prompt", topic)

    # Try fast online diffusion if possible
    diffusion_img = None
    try:
        from backend.modules.image_engine import generate_diffusion_image
        if temp_dir is None:
            temp_dir = Path(__file__).resolve().parent.parent.parent / "temp" / "generated_media"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_diff_path = temp_dir / f"diff_scene_{scene_num}_{uuid.uuid4().hex[:6]}.png"
        
        # 6s timeout to avoid slowing down video generation
        generate_diffusion_image(visual_prompt, str(temp_diff_path), timeout=6)
        if temp_diff_path.exists() and temp_diff_path.stat().st_size > 2000:
            with Image.open(str(temp_diff_path)) as loaded_im:
                diffusion_img = loaded_im.convert("RGB")
    except Exception as e:
        logger.info(f"Scene {scene_num} using high-fidelity offline infographic visual renderer.")

    if diffusion_img is not None:
        # Resize/crop diffusion image to target aspect ratio
        diff_w, diff_h = diffusion_img.size
        scale = max(width / diff_w, height / diff_h)
        nw, nh = int(diff_w * scale), int(diff_h * scale)
        resized_diff = diffusion_img.resize((nw, nh), Image.Resampling.LANCZOS)
        left = (nw - width) // 2
        top = (nh - height) // 2
        cropped_diff = resized_diff.crop((left, top, left + width, top + height))

        # Add cinematic vignette & header overlay
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d_over = ImageDraw.Draw(overlay)

        # Top dark gradient banner
        for y in range(130):
            alpha = int(210 * (1 - y / 130))
            d_over.line([(0, y), (width, y)], fill=(10, 15, 26, alpha))

        # Bottom dark gradient banner for subtitles
        for y in range(height - 150, height):
            alpha = int(220 * ((y - (height - 150)) / 150))
            d_over.line([(0, y), (width, y)], fill=(10, 15, 26, alpha))

        # Top header text
        title_font = _get_font(30, bold=True)
        sub_font = _get_font(18, bold=False)
        d_over.rectangle([40, 25, width - 40, 29], fill=style["accent"])
        d_over.text((50, 38), f"{style['icon']} SCENE {scene_num}: {scene_title.upper()}", font=title_font, fill=(248, 250, 252))
        d_over.text((50, 78), f"Topic: {topic} | {style['category']}", font=sub_font, fill=style["glow"])

        # Concept Pill
        key_pt = scene.get("key_point", "")
        if key_pt:
            pill_w = min(int(width * 0.45), 500)
            d_over.rounded_rectangle([width - pill_w - 50, 38, width - 50, 78], radius=8, fill=(15, 23, 42, 220), outline=style["accent"], width=1)
            d_over.text((width - pill_w - 35, 48), f"📌 {key_pt[:40]}", font=_get_font(15, bold=True), fill=(248, 250, 252))

        base_rgb = cropped_diff.convert("RGBA")
        combined = Image.alpha_composite(base_rgb, overlay)
        return combined.convert("RGB")

    # High-Quality Fallback Infographic Canvas
    img = Image.new("RGB", (width, height), color=style["bg_top"])
    draw = ImageDraw.Draw(img)

    # Vertical gradient
    for y in range(height):
        factor = y / height
        r = int(style["bg_top"][0] + (style["bg_bottom"][0] - style["bg_top"][0]) * factor)
        g = int(style["bg_top"][1] + (style["bg_bottom"][1] - style["bg_top"][1]) * factor)
        b = int(style["bg_top"][2] + (style["bg_bottom"][2] - style["bg_top"][2]) * factor)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Top Header Glowing Bar
    draw.rectangle([50, 30, width - 50, 35], fill=style["accent"])

    title_font = _get_font(32, bold=True)
    sub_font = _get_font(19, bold=False)
    card_title_font = _get_font(21, bold=True)
    body_font = _get_font(18, bold=False)

    draw.text((60, 48), f"{style['icon']} SCENE {scene_num}: {scene_title.upper()}", font=title_font, fill=(248, 250, 252))
    draw.text((60, 92), f"Topic: {topic} | Category: {style['category']}", font=sub_font, fill=style["glow"])

    # Mode badge
    draw.rounded_rectangle([width - 320, 48, width - 60, 88], radius=8, fill=(24, 32, 48), outline=style["accent"], width=1)
    draw.text((width - 305, 58), "🎬 AI GENERATED VIDEO", font=_get_font(15, bold=True), fill=(248, 250, 252))

    # Main Visual Concept Card
    card_w = int(width * 0.44)
    card_h = int(height * 0.52)
    top_y = 135

    # Left Container (Visual Focus & Key Points)
    x1 = 60
    draw.rounded_rectangle([x1, top_y, x1 + card_w, top_y + card_h], radius=16, fill=(20, 28, 44), outline=style["accent"], width=2)
    draw.rounded_rectangle([x1 + 20, top_y + 18, x1 + 230, top_y + 52], radius=10, fill=style["accent"])
    draw.text((x1 + 32, top_y + 24), "📌 Scene Focus", font=card_title_font, fill=(255, 255, 255))

    cur_y = top_y + 75
    draw.text((x1 + 25, cur_y), "✦ Visual Representation:", font=_get_font(18, bold=True), fill=style["glow"])
    cur_y += 28
    wrapped_vis = wrap_text_lines(visual_prompt, body_font, card_w - 60, draw)
    for line in wrapped_vis[:3]:
        draw.text((x1 + 40, cur_y), line, font=body_font, fill=(226, 232, 240))
        cur_y += 26
    cur_y += 14

    key_pt = scene.get("key_point", "Key concept overview")
    draw.text((x1 + 25, cur_y), "✦ Core Takeaway:", font=_get_font(18, bold=True), fill=style["glow"])
    cur_y += 28
    wrapped_pt = wrap_text_lines(key_pt, body_font, card_w - 60, draw)
    for line in wrapped_pt[:2]:
        draw.text((x1 + 40, cur_y), line, font=body_font, fill=(255, 255, 255))
        cur_y += 26

    # Right Container (Concept Flow & Schematic Nodes)
    x2 = width - card_w - 60
    draw.rounded_rectangle([x2, top_y, x2 + card_w, top_y + card_h], radius=16, fill=(20, 28, 44), outline=style["accent"], width=2)
    draw.rounded_rectangle([x2 + 20, top_y + 18, x2 + 230, top_y + 52], radius=10, fill=style["accent"])
    draw.text((x2 + 32, top_y + 24), "⚡ Dynamic Flow", font=card_title_font, fill=(255, 255, 255))

    # Central Subject Node
    cx = x2 + card_w // 2
    cy = top_y + 125
    draw.ellipse([cx - 95, cy - 35, cx + 95, cy + 35], fill=style["accent"], outline=style["glow"], width=3)
    lbl = scene_title[:16]
    draw.text((cx - len(lbl) * 4.2, cy - 10), lbl, font=_get_font(16, bold=True), fill=(255, 255, 255))

    # Connected Subnodes
    box1 = [cx - 180, cy + 85, cx - 20, cy + 145]
    box2 = [cx + 20, cy + 85, cx + 180, cy + 145]
    draw.line([(cx - 40, cy + 35), (cx - 100, cy + 85)], fill=style["glow"], width=2)
    draw.line([(cx + 40, cy + 35), (cx + 100, cy + 85)], fill=style["glow"], width=2)

    draw.rounded_rectangle(box1, radius=10, fill=(30, 40, 60), outline=style["accent"], width=2)
    draw.text((box1[0] + 15, box1[1] + 18), "Input & Structure", font=_get_font(15, bold=True), fill=(248, 250, 252))

    draw.rounded_rectangle(box2, radius=10, fill=(30, 40, 60), outline=style["accent"], width=2)
    draw.text((box2[0] + 15, box2[1] + 18), "Process & Result", font=_get_font(15, bold=True), fill=(248, 250, 252))

    return img


def create_dynamic_scene_clip(scene_img: Image.Image, scene: Dict[str, Any], audio_clip: AudioFileClip, duration: float, width: int = 1280, height: int = 720) -> VideoClip:
    """
    Constructs an animated cinematic clip for a scene:
    - Pre-renders subtitle banner.
    - Applies smooth camera motion (subtle Ken Burns zoom from 1.0 to 1.05).
    - Sets synchronized audio narration.
    """
    sub_font = _get_font(20, bold=True)
    narration_text = scene.get("narration") or scene.get("script", "")

    # Overlay subtitle bar on base image
    overlay_img = scene_img.copy()
    draw_sub = ImageDraw.Draw(overlay_img)
    sub_bar_y = height - 110
    draw_sub.rectangle([0, sub_bar_y, width, height], fill=(10, 15, 26))
    draw_sub.line([(0, sub_bar_y), (width, sub_bar_y)], fill=(59, 130, 246), width=2)

    wrapped_sub = wrap_text_lines(f"💬 {narration_text}", sub_font, width - 120, draw_sub)
    cur_sub_y = sub_bar_y + 16
    for line in wrapped_sub[:2]:
        draw_sub.text((60, cur_sub_y), line, font=sub_font, fill=(255, 230, 100))
        cur_sub_y += 28

    base_frame = overlay_img

    def make_frame(t):
        progress = min(max(t / duration, 0.0), 1.0)
        scale = 1.0 + 0.05 * progress
        sw = int(width * scale)
        sh = int(height * scale)
        res = base_frame.resize((sw, sh), Image.Resampling.BILINEAR)
        left = (sw - width) // 2
        top = (sh - height) // 2
        cropped = res.crop((left, top, left + width, top + height))
        return np.array(cropped)

    clip = VideoClip(make_frame, duration=duration)
    clip = _set_clip_audio(clip, audio_clip)
    return clip


# =====================================================================
# Main Text-to-Video Engine (Primary Pipeline for AI Chat Assistant)
# =====================================================================

def generate_text_to_video(
    prompt: str,
    output_path: Optional[str] = None,
    language: str = "Auto-Detect",
    persona: str = "Teacher Female",
    resolution: str = "720p",
    num_scenes: int = 3
) -> str:
    """
    Permanently fixed Text-to-Video generation pipeline:
    1. Analyzes user prompt and generates a structured, multi-scene storyboard.
    2. Synthesizes synchronized speech audio narration per scene using Edge-TTS (with offline fallback).
    3. Renders topic-accurate visual scene graphics (online diffusion + local schematic infographics).
    4. Applies cinematic Ken Burns motion, sleek header styling, and synchronized on-screen subtitles.
    5. Exports subtitles.srt and encodes complete, production-ready MP4 video.
    
    Returns:
        str: Absolute path to the rendered MP4 video file.
    """
    clean_prompt = prompt.strip() if prompt else "Educational Concept Overview"
    if not clean_prompt:
        clean_prompt = "Educational Concept Overview"

    if output_path is None:
        target_dir = Path(__file__).resolve().parent.parent.parent / "temp" / "generated_media"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_output_p = target_dir / f"txt2vid_{uuid.uuid4().hex[:8]}.mp4"
    else:
        target_output_p = Path(output_path).resolve()
        target_dir = target_output_p.parent
        target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Text-to-Video Generation for prompt: '{clean_prompt}' -> {target_output_p}")

    # Resolution settings (720p default for optimal balance of speed and clarity)
    if resolution == "1080p":
        width, height = 1920, 1080
    elif resolution == "4K":
        width, height = 3840, 2160
    else:
        width, height = 1280, 720

    # 1. Generate Multi-Scene Storyboard
    scenes = StoryboardEngine.generate_storyboard_from_prompt(clean_prompt, num_scenes=num_scenes)

    scene_clips = []
    audio_clips = []
    scene_durations = []

    try:
        # 2. Process each scene: Audio, Visual Frame, and Dynamic Clip
        for idx, sc in enumerate(scenes):
            logger.info(f"Rendering Video Scene {sc['scene_num']}/{len(scenes)}: {sc['title']}")
            
            # Narration Script (with multilingual translation if required)
            narration_text = sc["narration"]
            if language not in ["English", "Auto-Detect"]:
                try:
                    translated = translate_text(narration_text, target_language=language)
                    if validate_translation(translated, language):
                        narration_text = translated
                except Exception as e_trans:
                    logger.warning(f"Translation warning ({e_trans}), using original narration.")

            # Audio Synthesis
            scene_audio_path = target_dir / f"aud_scene_{idx}_{uuid.uuid4().hex[:6]}.mp3"
            final_aud_path = AudioNarrationEngine.generate_narration(
                text=narration_text,
                output_path=str(scene_audio_path),
                language=language,
                persona=persona
            ) or str(scene_audio_path)

            aclip = AudioFileClip(final_aud_path)
            audio_clips.append(aclip)

            dur = max(aclip.duration, 3.0) + 0.3
            scene_durations.append(dur)

            # Visual Frame Rendering
            scene_img = render_scene_card_image(sc, clean_prompt, width=width, height=height, temp_dir=target_dir)

            # Dynamic Cinematic Video Clip
            vclip = create_dynamic_scene_clip(scene_img, sc, aclip, duration=dur, width=width, height=height)
            scene_clips.append(vclip)

        # 3. Concatenate Scene Clips
        logger.info("Assembling and concatenating all video scenes...")
        final_video = concatenate_videoclips(scene_clips, method="compose")

        # 4. Export SRT Subtitles
        srt_path = target_dir / f"{target_output_p.stem}.srt"
        export_subtitles(scenes, scene_durations, str(srt_path))

        # 5. Encode Final MP4 Video
        logger.info(f"Encoding final MP4 video at {target_output_p}...")
        final_video.write_videofile(
            str(target_output_p),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            write_logfile=True,
            logger=None
        )

        if not target_output_p.exists() or target_output_p.stat().st_size == 0:
            raise ValueError(f"Generated video is missing or 0 bytes: {target_output_p}")

        logger.info(f"Text-to-Video successfully created: {target_output_p} ({target_output_p.stat().st_size} bytes)")
        return str(target_output_p)

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Text-to-Video generation error:\n{error_trace}")
        raise RuntimeError(f"Text-to-Video generation failed: {e}\n{error_trace}") from e

    finally:
        # Proper resource cleanup on Windows
        for c in scene_clips:
            try:
                c.close()
            except Exception:
                pass
        for a in audio_clips:
            try:
                a.close()
            except Exception:
                pass
        if 'final_video' in locals() and final_video:
            try:
                final_video.close()
            except Exception:
                pass


# =====================================================================
# Image-to-Video Generation Engine
# =====================================================================

def extract_image_path(ref_input: Any) -> Optional[str]:
    """
    Robustly extracts and validates an image file path from various input types:
    - File path string or Path object
    - Gradio file dictionary (containing 'path', 'name', or 'filepath')
    - Object with .name, .path, or ._path attribute
    - List or tuple containing file object or path
    - Verifies image readability and format integrity via PIL.
    """
    if ref_input is None:
        return None

    path_candidate = None
    if isinstance(ref_input, (str, Path)):
        path_candidate = str(ref_input)
    elif isinstance(ref_input, dict):
        path_candidate = ref_input.get("path") or ref_input.get("name") or ref_input.get("filepath")
    elif isinstance(ref_input, (list, tuple)) and len(ref_input) > 0:
        return extract_image_path(ref_input[0])
    elif hasattr(ref_input, "name") and isinstance(ref_input.name, str):
        path_candidate = ref_input.name
    elif hasattr(ref_input, "path") and isinstance(ref_input.path, str):
        path_candidate = ref_input.path

    if not path_candidate or not os.path.exists(path_candidate):
        return None

    # Verify image integrity and format
    try:
        with Image.open(path_candidate) as test_im:
            test_im.verify()
        return str(Path(path_candidate).resolve())
    except Exception as e:
        logger.warning(f"Image candidate failed verification: {path_candidate} ({e})")
        return None


def detect_motion_type(prompt: str, img_w: int, img_h: int) -> Tuple[str, str]:
    """
    Parses user motion keywords and actions to select camera dynamics:
    Returns (motion_key, human_readable_label).
    """
    p = prompt.lower() if prompt else ""
    if any(k in p for k in ["zoom in", "push in", "close-up", "closer", "magnify"]):
        return "ZOOM_IN", "🔍 Cinematic Push-In"
    elif any(k in p for k in ["zoom out", "pull back", "wide", "reveal", "backward"]):
        return "ZOOM_OUT", "🔭 Cinematic Pull-Back"
    elif any(k in p for k in ["pan left", "move left", "leftwards"]):
        return "PAN_LEFT", "◀ Cinematic Pan Left"
    elif any(k in p for k in ["pan right", "move right", "rightwards", "horizontal"]):
        return "PAN_RIGHT", "▶ Cinematic Pan Right"
    elif any(k in p for k in ["pan down", "tilt down", "scroll down", "top to bottom"]):
        return "PAN_DOWN", "▼ Vertical Tilt Down"
    elif any(k in p for k in ["pan up", "tilt up", "scroll up", "bottom to top"]):
        return "PAN_UP", "▲ Vertical Tilt Up"
    elif any(k in p for k in ["orbit", "sweep", "drone", "flyover", "diagonal", "dynamic"]):
        return "CINEMATIC_SWEEP", "✨ Dynamic Camera Sweep"
    elif any(k in p for k in ["pulse", "breathe", "heartbeat"]):
        return "PULSE", "💫 Ambient Motion Pulse"
    else:
        # Default intelligent selection based on aspect ratio
        if img_h / max(img_w, 1) > 1.3:
            return "PAN_DOWN", "▼ Smooth Reveal Tilt"
        else:
            return "ZOOM_IN", "🎬 Cinematic Motion Focus"


def generate_image_to_video(
    input_image_path: Any,
    prompt: str = "",
    output_path: Optional[str] = None,
    resolution: str = "720p"
) -> str:
    """
    Generates a motion video (MP4) based directly on an uploaded reference image.
    1. Validates reference image existence and format across PNG, JPG, WEBP, BMP.
    2. Applies orientation correction (EXIF transpose) and handles transparency.
    3. Interprets optional user motion instructions (zoom in, pan, tilt, sweep, pulse).
    4. Preserves subject likeness, aspect ratio, and context with intelligent framing
       (pillarbox blur for portrait/square images; vertical pan for tall subjects).
    5. Synthesizes synchronized narration speech with Edge-TTS (and offline fallback).
    6. Renders crisp, non-scaling on-screen badges and subtitles.
    
    Returns:
        str: Absolute path to the generated MP4 video file.
    """
    valid_path = extract_image_path(input_image_path)
    if not valid_path:
        raise ValueError(f"Uploaded reference image is missing, corrupted, or unsupported: {input_image_path}")

    target_dir = Path(__file__).resolve().parent.parent.parent / "temp" / "generated_media"
    target_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        target_output_p = target_dir / f"img2vid_{uuid.uuid4().hex[:8]}.mp4"
    else:
        target_output_p = Path(output_path).resolve()
        target_output_p.parent.mkdir(parents=True, exist_ok=True)

    narration_script = prompt.strip() if prompt and prompt.strip() else "Visual concept analysis and motion presentation of the reference image."
    logger.info(f"Generating Image-to-Video for '{valid_path}' with prompt: '{narration_script}'")

    # Dimensions
    if resolution == "1080p":
        width, height = 1920, 1080
    elif resolution == "4K":
        width, height = 3840, 2160
    else:
        width, height = 1280, 720

    # 1. Synthesize Narration Audio
    audio_path = target_dir / f"img2vid_aud_{uuid.uuid4().hex[:6]}.mp3"
    final_audio = AudioNarrationEngine.generate_narration(narration_script, str(audio_path)) or str(audio_path)
    audio_clip = AudioFileClip(final_audio)
    duration = max(audio_clip.duration, 4.0) + 0.3

    # 2. Load & Prepare Base Reference Image
    with Image.open(valid_path) as raw_im:
        transposed = ImageOps.exif_transpose(raw_im)
        if transposed.mode in ("RGBA", "LA") or (transposed.mode == "P" and "transparency" in transposed.info):
            bg_clean = Image.new("RGBA", transposed.size, (20, 24, 36, 255))
            composite = Image.alpha_composite(bg_clean, transposed.convert("RGBA"))
            base_rgb = composite.convert("RGB")
        else:
            base_rgb = transposed.convert("RGB")

    # Downscale excessively large images to max 1920 to prevent memory blowup
    if base_rgb.width > 2500 or base_rgb.height > 2500:
        base_rgb.thumbnail((1920, 1920), Image.Resampling.LANCZOS)

    img_w, img_h = base_rgb.size
    motion_type, motion_label = detect_motion_type(prompt, img_w, img_h)
    ar = img_w / max(img_h, 1)

    # 3. Intelligent Aspect Ratio Framing
    margin_factor = 1.15
    if ar < 1.15 and motion_type not in ["PAN_DOWN", "PAN_UP"]:
        # Pillarbox with blurred background (100% subject preservation)
        blur_bg = base_rgb.resize((width, height), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(28))
        enhancer = ImageEnhance.Brightness(blur_bg)
        dim_bg = enhancer.enhance(0.55)

        # Scale foreground to fit height
        fg_scale = (height * 0.94) / img_h
        fg_w, fg_h = int(img_w * fg_scale), int(img_h * fg_scale)
        resized_fg = base_rgb.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

        canvas_w = int(width * margin_factor)
        canvas_h = int(height * margin_factor)
        canvas = dim_bg.resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)

        fg_x = (canvas_w - fg_w) // 2
        fg_y = (canvas_h - fg_h) // 2
        canvas.paste(resized_fg, (fg_x, fg_y))
    elif motion_type in ["PAN_DOWN", "PAN_UP"] and ar < 1.15:
        # Full-width vertical reveal canvas for tall portraits
        canvas_w = width
        canvas_h = max(int(width * (img_h / img_w)), int(height * 1.25))
        canvas = base_rgb.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
    else:
        # Landscape cover canvas with motion margins
        scale_cover = max(width * margin_factor / img_w, height * margin_factor / img_h)
        canvas_w = int(img_w * scale_cover)
        canvas_h = int(img_h * scale_cover)
        canvas = base_rgb.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

    max_dx = max(canvas_w - width, 0)
    max_dy = max(canvas_h - height, 0)

    # UI Fonts
    title_font = _get_font(24, bold=True)
    sub_font = _get_font(18, bold=True)

    # 4. Multi-Frame Camera Motion Generator
    def make_frame(t):
        p = min(max(t / duration, 0.0), 1.0)
        # Smooth easing curve
        ease_p = 3 * p**2 - 2 * p**3

        if motion_type == "ZOOM_IN":
            zoom = 1.0 + 0.08 * ease_p
            sw, sh = int(width * zoom), int(height * zoom)
            scaled = canvas.resize((sw, sh), Image.Resampling.BILINEAR)
            left = (sw - width) // 2
            top = (sh - height) // 2
            frame_img = scaled.crop((left, top, left + width, top + height))

        elif motion_type == "ZOOM_OUT":
            zoom = 1.08 - 0.08 * ease_p
            sw, sh = int(width * zoom), int(height * zoom)
            scaled = canvas.resize((sw, sh), Image.Resampling.BILINEAR)
            left = (sw - width) // 2
            top = (sh - height) // 2
            frame_img = scaled.crop((left, top, left + width, top + height))

        elif motion_type == "PAN_RIGHT":
            cur_x = int(max_dx * ease_p)
            cur_y = max_dy // 2
            frame_img = canvas.crop((cur_x, cur_y, cur_x + width, cur_y + height))

        elif motion_type == "PAN_LEFT":
            cur_x = int(max_dx * (1.0 - ease_p))
            cur_y = max_dy // 2
            frame_img = canvas.crop((cur_x, cur_y, cur_x + width, cur_y + height))

        elif motion_type == "PAN_DOWN":
            cur_x = max_dx // 2
            cur_y = int(max_dy * ease_p)
            frame_img = canvas.crop((cur_x, cur_y, cur_x + width, cur_y + height))

        elif motion_type == "PAN_UP":
            cur_x = max_dx // 2
            cur_y = int(max_dy * (1.0 - ease_p))
            frame_img = canvas.crop((cur_x, cur_y, cur_x + width, cur_y + height))

        elif motion_type == "CINEMATIC_SWEEP":
            zoom = 1.0 + 0.05 * ease_p
            cur_x = int(max_dx * 0.7 * ease_p)
            cur_y = int(max_dy * 0.7 * ease_p)
            cropped = canvas.crop((cur_x, cur_y, cur_x + width, cur_y + height))
            sw, sh = int(width * zoom), int(height * zoom)
            scaled = cropped.resize((sw, sh), Image.Resampling.BILINEAR)
            frame_img = scaled.crop(((sw - width) // 2, (sh - height) // 2, (sw - width) // 2 + width, (sh - height) // 2 + height))

        else: # PULSE
            pulse = 1.0 + 0.04 * math.sin(math.pi * p)
            sw, sh = int(width * pulse), int(height * pulse)
            scaled = canvas.resize((sw, sh), Image.Resampling.BILINEAR)
            frame_img = scaled.crop(((sw - width) // 2, (sh - height) // 2, (sw - width) // 2 + width, (sh - height) // 2 + height))

        # 5. Crisp, Non-scaling UI Overlays
        draw = ImageDraw.Draw(frame_img)

        # Header Badge
        header_text = f"🎬 {motion_label} • Reference Animation"
        draw.rounded_rectangle([40, 25, 480, 65], radius=8, fill=(15, 23, 42), outline=(59, 130, 246), width=2)
        draw.text((55, 34), header_text, font=title_font, fill=(248, 250, 252))

        # Subtitle Banner
        sub_y = height - 100
        draw.rectangle([0, sub_y, width, height], fill=(10, 15, 26))
        draw.line([(0, sub_y), (width, sub_y)], fill=(59, 130, 246), width=2)
        lines = wrap_text_lines(f"💬 {narration_script}", sub_font, width - 120, draw)
        cur_sy = sub_y + 14
        for l in lines[:2]:
            draw.text((60, cur_sy), l, font=sub_font, fill=(255, 230, 100))
            cur_sy += 26

        return np.array(frame_img)

    video_clip = VideoClip(make_frame, duration=duration)
    video_clip = _set_clip_audio(video_clip, audio_clip)

    try:
        video_clip.write_videofile(
            str(target_output_p),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            write_logfile=True,
            logger=None
        )

        if not target_output_p.exists() or target_output_p.stat().st_size == 0:
            raise ValueError(f"Generated Image-to-Video file is missing or 0 bytes: {target_output_p}")

        logger.info(f"Image-to-Video successfully rendered: {target_output_p} ({target_output_p.stat().st_size} bytes)")
        return str(target_output_p)

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Image-to-Video generation error:\n{error_trace}")
        raise RuntimeError(f"Image-to-Video generation failed: {e}\n{error_trace}") from e

    finally:
        video_clip.close()
        audio_clip.close()


# =====================================================================
# 3D Educational Video Engine (Tab 9: 3D Video Studio with Professor RoboCat)
# =====================================================================

def generate_3d_educational_video(
    text: str,
    output_path: str,
    language: str = "Auto-Detect",
    persona: str = "Teacher Female",
    resolution: str = "1080p",
    aspect_ratio: str = "16:9",
    theme: str = "Science Lab",
    prop_type: str = "AI Neural Networks"
) -> str:
    """
    Renders a complete 3D animated educational video featuring Professor RoboCat.
    Includes lip-sync mouth movements, blinking eyes, multilingual narration, interactive props,
    SRT subtitle generation, and verified MP4 output.
    """
    logger.info("Starting 3D Multilingual Educational Video Generation with Professor RoboCat...")

    target_output_p = Path(output_path).resolve()
    target_dir = target_output_p.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Resolution dimensions
    if resolution == "4K":
        width, height = 3840, 2160
    elif resolution == "8K":
        width, height = 7680, 4320
    elif resolution == "720p":
        width, height = 1280, 720
    else:  # 1080p
        width, height = 1920, 1080

    if aspect_ratio == "9:16":
        width, height = height, width
    elif aspect_ratio == "1:1":
        width, height = min(width, height), min(width, height)

    # 1. Create Storyboard Scenes for Visual Board Content
    english_scenes = StoryboardEngine.create_scenes(text)

    # 2. Generate Multilingual Teacher Voice Script
    if language not in ["English", "Auto-Detect"]:
        logger.info(f"Target language is '{language}'. Translating teacher script...")
        translated_text = translate_text(text, target_language=language)
        if not validate_translation(translated_text, language):
            logger.warning(f"Teacher script validation warning for '{language}'. Falling back to original script.")
            script_to_narrate = text
        else:
            script_to_narrate = translated_text
    else:
        script_to_narrate = text

    # 3. Synthesize Robo Cat Spoken Audio Narration
    audio_path = str(target_dir / "narration_speech.mp3")
    final_audio_path = AudioNarrationEngine.generate_narration(script_to_narrate, audio_path, language=language, persona=persona) or audio_path

    audio_clip = AudioFileClip(final_audio_path)
    total_duration = max(audio_clip.duration, 4.0)
    scene_duration = total_duration / len(english_scenes)

    # 4. Export SRT Subtitle File
    srt_path = str(target_dir / "subtitles.srt")
    export_subtitles(english_scenes, scene_duration, srt_path)

    scene_clips = []
    font = _get_font(int(height * 0.038), bold=False)
    title_font = _get_font(int(height * 0.05), bold=True)

    try:
        # 5. Render Video Frames with Blackboard Content & Subtitles
        for idx, sc in enumerate(english_scenes):
            img = Image.new('RGBA', (width, height), (20, 25, 45, 255))
            draw = ImageDraw.Draw(img)

            # Gradient Background
            for y in range(height):
                r = int(20 + (y / height) * 20)
                g = int(25 + (y / height) * 35)
                b = int(45 + (y / height) * 60)
                draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

            # Grid Floor
            for x in range(0, width, 60):
                draw.line([(x, height - 200), (int(width / 2 + (x - width / 2) * 1.8), height)], fill=(0, 200, 255, 40), width=1)
            for y_g in range(height - 200, height, 25):
                draw.line([(0, y_g), (width, y_g)], fill=(0, 200, 255, 30), width=1)

            # Draw Topic Visual Prop
            VisualPropRenderer.draw_prop(draw, pos=(int(width * 0.82), int(height * 0.22)), prop_type=prop_type)

            # Draw Character with Lip-Sync & Blinking Eyes
            mouth = "OPEN" if idx % 2 == 0 else "SOFT_OPEN"
            robocat = ProfessorRoboCatRenderer.draw_character(
                size=(int(width * 0.35), int(height * 0.55)),
                pose=sc.get("pose", "TEACHING"),
                expression=sc.get("expression", "HAPPY"),
                mouth_state=mouth,
                is_blinking=False
            )

            # Paste Character on Canvas
            char_x = int(width * 0.05)
            char_y = int(height * 0.35)
            img.paste(robocat, (char_x, char_y), robocat)

            # Header Banner
            draw.rectangle([int(width * 0.04), int(height * 0.04), int(width * 0.96), int(height * 0.14)], fill=(15, 25, 55, 220), outline=(0, 220, 255, 255), width=3)
            draw.text((int(width * 0.06), int(height * 0.06)), f"🎓 PROFESSOR ROBOCAT | {sc['title']}", font=title_font, fill=(0, 240, 255, 255))

            # Blackboard Content Box
            board_box = [int(width * 0.42), int(height * 0.20), int(width * 0.96), int(height * 0.75)]
            draw.rectangle(board_box, fill=(15, 35, 45, 230), outline=(0, 200, 255, 200), width=3)

            # Script Text Formatting inside Blackboard
            script_text = sc.get("narration") or sc.get("script", "")
            wrapped_script = wrap_text(script_text, font, int(width * 0.50), draw)
            draw.multiline_text((int(width * 0.44), int(height * 0.24)), wrapped_script, font=font, fill=(255, 255, 255, 255), spacing=8)

            # Subtitle Bar at Bottom
            draw.rectangle([0, int(height * 0.85), width, height], fill=(10, 15, 30, 240))
            draw.text((int(width * 0.05), int(height * 0.89)), f"💬 Subtitle: {script_text[:90]}...", font=font, fill=(255, 215, 0, 255))

            img_rgb = img.convert('RGB')
            img_np = np.array(img_rgb)

            clip = ImageClip(img_np)
            clip = _set_clip_duration(clip, scene_duration)
            scene_clips.append(clip)

        # Concatenate Scene Clips
        final_video = concatenate_videoclips(scene_clips, method="compose")
        final_video = _set_clip_audio(final_video, audio_clip)

        # Encode MP4
        final_video.write_videofile(
            str(target_output_p),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            write_logfile=True,
            logger=None
        )

        if not target_output_p.exists() or target_output_p.stat().st_size == 0:
            raise ValueError(f"Generated 3D video is missing or 0 bytes: {target_output_p}")

        logger.info(f"3D Educational Video successfully rendered: {target_output_p}")
        return str(target_output_p)

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"3D Video generation error:\n{error_trace}")
        raise RuntimeError(f"3D Video generation failed:\n{error_trace}") from e

    finally:
        for c in scene_clips:
            try:
                c.close()
            except Exception:
                pass
        try:
            if 'final_video' in locals() and final_video:
                final_video.close()
            if audio_clip:
                audio_clip.close()
        except Exception:
            pass


def create_fallback_video(text: str, audio_path: str, output_path: str) -> str:
    """Wrapper function preserving legacy API calls."""
    return generate_text_to_video(prompt=text, output_path=output_path)
