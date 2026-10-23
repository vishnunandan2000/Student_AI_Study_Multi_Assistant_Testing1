import os
import asyncio
import logging
import traceback
import edge_tts
from pathlib import Path
from typing import Dict, Any, Optional
from backend.modules.translator import detect_language, validate_translation, UNICODE_RANGES

# Configure module logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Expanded 30+ Language Voice Mapping
MULTILINGUAL_VOICES: Dict[str, Dict[str, str]] = {
    "English": {"Teacher Female": "en-US-AriaNeural", "Teacher Male": "en-US-GuyNeural", "Cartoon Child": "en-US-AnaNeural"},
    "Hindi": {"Teacher Female": "hi-IN-SwaraNeural", "Teacher Male": "hi-IN-MadhurNeural", "Cartoon Child": "hi-IN-SwaraNeural"},
    "Telugu": {"Teacher Female": "te-IN-ShrutiNeural", "Teacher Male": "te-IN-MohanNeural", "Cartoon Child": "te-IN-ShrutiNeural"},
    "Kannada": {"Teacher Female": "kn-IN-SapnaNeural", "Teacher Male": "kn-IN-GaganNeural", "Cartoon Child": "kn-IN-SapnaNeural"},
    "Tamil": {"Teacher Female": "ta-IN-PallaviNeural", "Teacher Male": "ta-IN-ValluvarNeural", "Cartoon Child": "ta-IN-PallaviNeural"},
    "Malayalam": {"Teacher Female": "ml-IN-SobhanaNeural", "Teacher Male": "ml-IN-MidhunNeural", "Cartoon Child": "ml-IN-SobhanaNeural"},
    "Marathi": {"Teacher Female": "mr-IN-AarohiNeural", "Teacher Male": "mr-IN-ManoharNeural", "Cartoon Child": "mr-IN-AarohiNeural"},
    "Gujarati": {"Teacher Female": "gu-IN-DhwaniNeural", "Teacher Male": "gu-IN-NiranjanNeural", "Cartoon Child": "gu-IN-DhwaniNeural"},
    "Bengali": {"Teacher Female": "bn-IN-TanishaaNeural", "Teacher Male": "bn-IN-BashkarNeural", "Cartoon Child": "bn-IN-TanishaaNeural"},
    "Punjabi": {"Teacher Female": "pa-IN-GurpreetNeural", "Teacher Male": "pa-IN-GurpreetNeural", "Cartoon Child": "pa-IN-GurpreetNeural"},
    "Urdu": {"Teacher Female": "ur-PK-UzmaNeural", "Teacher Male": "ur-PK-AsadNeural", "Cartoon Child": "ur-PK-UzmaNeural"},
    "French": {"Teacher Female": "fr-FR-DeniseNeural", "Teacher Male": "fr-FR-HenriNeural", "Cartoon Child": "fr-FR-EloiseNeural"},
    "German": {"Teacher Female": "de-DE-KatjaNeural", "Teacher Male": "de-DE-KillianNeural", "Cartoon Child": "de-DE-LouisaNeural"},
    "Spanish": {"Teacher Female": "es-ES-ElviraNeural", "Teacher Male": "es-ES-AlvaroNeural", "Cartoon Child": "es-ES-AbrilNeural"},
    "Portuguese": {"Teacher Female": "pt-BR-FranciscaNeural", "Teacher Male": "pt-BR-AntonioNeural", "Cartoon Child": "pt-BR-FranciscaNeural"},
    "Italian": {"Teacher Female": "it-IT-ElsaNeural", "Teacher Male": "it-IT-DiegoNeural", "Cartoon Child": "it-IT-IsabellaNeural"},
    "Japanese": {"Teacher Female": "ja-JP-NanamiNeural", "Teacher Male": "ja-JP-KeitaNeural", "Cartoon Child": "ja-JP-AoiNeural"},
    "Chinese": {"Teacher Female": "zh-CN-XiaoxiaoNeural", "Teacher Male": "zh-CN-YunjianNeural", "Cartoon Child": "zh-CN-XiaoxiaoNeural"},
    "Korean": {"Teacher Female": "ko-KR-SunHiNeural", "Teacher Male": "ko-KR-InJoonNeural", "Cartoon Child": "ko-KR-SunHiNeural"},
    "Arabic": {"Teacher Female": "ar-SA-ZariyahNeural", "Teacher Male": "ar-SA-HamedNeural", "Cartoon Child": "ar-SA-ZariyahNeural"},
    "Russian": {"Teacher Female": "ru-RU-SvetlanaNeural", "Teacher Male": "ru-RU-DmitryNeural", "Cartoon Child": "ru-RU-SvetlanaNeural"},
    "Turkish": {"Teacher Female": "tr-TR-EmelNeural", "Teacher Male": "tr-TR-AhmetNeural", "Cartoon Child": "tr-TR-EmelNeural"},
    "Dutch": {"Teacher Female": "nl-NL-ColetteNeural", "Teacher Male": "nl-NL-MaartenNeural", "Cartoon Child": "nl-NL-FennaNeural"},
    "Thai": {"Teacher Female": "th-TH-PremwadeeNeural", "Teacher Male": "th-TH-NiwatNeural", "Cartoon Child": "th-TH-PremwadeeNeural"},
    "Vietnamese": {"Teacher Female": "vi-VN-HoaiMyNeural", "Teacher Male": "vi-VN-NamMinhNeural", "Cartoon Child": "vi-VN-HoaiMyNeural"},
    "Indonesian": {"Teacher Female": "id-ID-GadisNeural", "Teacher Male": "id-ID-ArdiNeural", "Cartoon Child": "id-ID-GadisNeural"},
    "Malay": {"Teacher Female": "ms-MY-YasminNeural", "Teacher Male": "ms-MY-OsmanNeural", "Cartoon Child": "ms-MY-YasminNeural"},
}


def detect_language(text: str) -> str:
    """Detect language based on Unicode script ranges."""
    for char in text:
        code = ord(char)
        if 0x0C00 <= code <= 0x0C7F:
            return "Telugu"
        elif 0x0900 <= code <= 0x097F:
            return "Hindi"
        elif 0x0C80 <= code <= 0x0CFF:
            return "Kannada"
        elif 0x0B80 <= code <= 0x0BFF:
            return "Tamil"
        elif 0x0D00 <= code <= 0x0D7F:
            return "Malayalam"
        elif 0x0A00 <= code <= 0x0A7F:
            return "Punjabi"
        elif 0x0980 <= code <= 0x09FF:
            return "Bengali"
        elif 0x0B00 <= code <= 0x0B7F:
            return "Odia"
        elif 0x0600 <= code <= 0x06FF:
            return "Arabic"
        elif 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF:
            return "Japanese"
        elif 0xAC00 <= code <= 0xD7AF:
            return "Korean"
        elif 0x0400 <= code <= 0x04FF:
            return "Russian"
    return "English"


async def generate_audio_async(text: str, output_path: str, voice: str = "en-US-AriaNeural") -> str:
    """Asynchronously synthesize speech audio using Edge-TTS."""
    target_p = Path(output_path).resolve()
    logger.info(f"Synthesizing audio with voice: {voice}")

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(target_p))

    if not target_p.exists() or target_p.stat().st_size == 0:
        raise ValueError(f"Generated audio file is missing or 0 bytes: {target_p}")

    return str(target_p)


def generate_audio_offline(text: str, output_path: str, persona: str = "Teacher Female") -> str:
    """
    100% Offline fallback speech synthesis using Windows SAPI5 / pyttsx3.
    Ensures narration and video generation never fail even if internet or Edge-TTS is down.
    """
    target_p = Path(output_path).resolve()
    target_p.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using Offline TTS Fallback (pyttsx3) for: {target_p}")

    co_initialized = False
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
            co_initialized = True
        except ImportError:
            pass

        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)

        # Select matching gender voice from Windows SAPI5 if available
        voices = engine.getProperty("voices")
        if voices:
            selected_voice = None
            if persona in ["Teacher Female", "Cartoon Child"]:
                for v in voices:
                    if "zira" in v.name.lower() or "female" in v.name.lower():
                        selected_voice = v.id
                        break
            elif persona == "Teacher Male":
                for v in voices:
                    if "david" in v.name.lower() or "male" in v.name.lower():
                        selected_voice = v.id
                        break
            if selected_voice:
                engine.setProperty("voice", selected_voice)

        engine.save_to_file(text, str(target_p))
        engine.runAndWait()

        if target_p.exists() and target_p.stat().st_size > 0:
            logger.info(f"Offline TTS successfully generated: {target_p} ({target_p.stat().st_size} bytes)")
            return str(target_p)
        else:
            raise ValueError(f"pyttsx3 generated 0 bytes file: {target_p}")

    except Exception as e:
        logger.error(f"Offline TTS error: {e}")
        # Last-resort emergency fallback: generate a valid silent audio file so video pipelines don't crash
        try:
            import wave
            wav_path = target_p.with_suffix(".wav")
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                # 3 seconds of silence
                wf.writeframes(b"\x00" * (22050 * 2 * 3))
            logger.warning(f"Created emergency silent audio placeholder at: {wav_path}")
            return str(wav_path)
        except Exception as e_wave:
            raise RuntimeError(f"Both online Edge-TTS and offline TTS fallback failed: {e}\nEmergency wave error: {e_wave}")
    finally:
        if co_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass


def generate_audio(
    text: str,
    output_path: str,
    voice: str = "en-US-AriaNeural",
    language: str = "Auto-Detect",
    persona: str = "Teacher Female"
) -> str:
    """
    Speech synthesis with automatic offline fallback.
    Tries high-quality online Edge-TTS first; falls back seamlessly to offline Windows TTS (pyttsx3) on error.
    """
    target_p = Path(output_path).resolve()
    target_p.parent.mkdir(parents=True, exist_ok=True)

    if language == "Auto-Detect":
        detected_lang = detect_language(text)
        logger.info(f"Auto-detected script language: '{detected_lang}'")
        selected_voice = MULTILINGUAL_VOICES.get(detected_lang, {}).get(persona, "en-US-AriaNeural")
    else:
        selected_voice = MULTILINGUAL_VOICES.get(language, {}).get(persona, voice)
        # Pre-flight Validation: Ensure text matches requested language script
        if language in UNICODE_RANGES and not validate_translation(text, language):
            logger.warning(
                f"Validation Alert: Synthesizing narration for '{language}' using voice '{selected_voice}'. "
                f"Input text snippet: '{text[:50]}...'"
            )

    logger.info(f"Generating narration audio for '{target_p}' using voice '{selected_voice}'")

    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(generate_audio_async(text, str(target_p), selected_voice))
        return result
    except Exception as e:
        logger.warning(
            f"⚠️ Online Edge-TTS synthesis failed ({e}). "
            f"Seamlessly switching to offline Windows TTS (pyttsx3)..."
        )
        return generate_audio_offline(text, str(target_p), persona=persona)
    finally:
        if loop is not None:
            try:
                loop.close()
            except Exception:
                pass


class AudioNarrationEngine:
    """Class wrapper for narration audio synthesis."""
    @staticmethod
    def generate_narration(
        text: str,
        output_path: str,
        voice: str = "en-US-AriaNeural",
        language: str = "Auto-Detect",
        persona: str = "Teacher Female"
    ) -> str:
        return generate_audio(text, output_path, voice=voice, language=language, persona=persona)




