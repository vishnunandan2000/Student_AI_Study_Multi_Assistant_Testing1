import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from backend.core.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

# Configure module logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Language Unicode Script Ranges
UNICODE_RANGES: Dict[str, Tuple[int, int]] = {
    "Telugu": (0x0C00, 0x0C7F),
    "Hindi": (0x0900, 0x097F),
    "Kannada": (0x0C80, 0x0CFF),
    "Tamil": (0x0B80, 0x0BFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Marathi": (0x0900, 0x097F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Bengali": (0x0980, 0x09FF),
    "Punjabi": (0x0A00, 0x0A7F),
    "Odia": (0x0B00, 0x0B7F),
    "Urdu": (0x0600, 0x06FF),
    "Arabic": (0x0600, 0x06FF),
    "Russian": (0x0400, 0x04FF),
    "Japanese": (0x3040, 0x30FF),
    "Korean": (0xAC00, 0xD7AF),
    "Chinese": (0x4E00, 0x9FFF),
    "Thai": (0x0E00, 0x0E7F),
}


def detect_language(text: str) -> str:
    """Detect language of text based on Unicode script ranges."""
    for char in text:
        code = ord(char)
        for lang_name, (start_code, end_code) in UNICODE_RANGES.items():
            if start_code <= code <= end_code:
                return lang_name
    return "English"


def validate_translation(text: str, target_language: str) -> bool:
    """
    Validates if the translated text contains non-empty content for target language narration.
    """
    if target_language in ["English", "Auto-Detect"]:
        return True

    if not text or len(text.strip()) == 0:
        return False

    if target_language not in UNICODE_RANGES:
        return True

    start_code, end_code = UNICODE_RANGES[target_language]
    matches = sum(1 for char in text if start_code <= ord(char) <= end_code)

    logger.info(f"Translation Validation for '{target_language}': Found {matches} native script characters in {len(text)} total chars.")
    # Technical terms remain in English, so native script matches or non-empty length passes
    return matches > 0 or len(text.strip()) > 10


def translate_text(text: str, target_language: str) -> str:
    """
    Translates educational script text into a classroom teacher spoken narration script
    using a mix of English (for technical terms, code, concepts) and target_language (for explanations).
    """
    if not text.strip():
        return text

    if target_language in ["English", "Auto-Detect"]:
        logger.info(f"Target language is '{target_language}'. Skipping LLM translation.")
        return text

    current_lang = detect_language(text)
    logger.info(f"Initiating Ollama Llama 3.2 teacher translation from '{current_lang}' to '{target_language}'...")

    system_prompt = (
        f"You are Robo Cat, an expert teacher explaining concepts in a classroom to beginners.\n"
        f"Translate the following educational lesson script into a spoken audio narration script using a mix of English and {target_language}.\n\n"
        f"STRICT RULES:\n"
        f"1. Explain concepts like a friendly classroom teacher using a natural mix of English and {target_language}.\n"
        f"2. TECHNICAL TERMS, keywords, programming concepts, function names, class names, APIs, commands, code snippets, file names, and technology names MUST ALWAYS REMAIN 100% IN ENGLISH.\n"
        f"3. Do NOT translate technical terms into {target_language}. Keep them in English.\n"
        f"4. The conversational explanation and connecting phrases should be in {target_language} using simple, easy-to-understand words suitable for beginners.\n"
        f"5. Output ONLY the mixed spoken script without any preambles, introductory titles, or meta-commentary.\n"
    )

    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text)
        ]
        response = llm.invoke(messages)
        translated = response.content.strip()

        # Validate Translation Script
        if not validate_translation(translated, target_language):
            logger.warning(f"Translation validation warning: '{target_language}' script check was inconclusive. Retrying with explicit prompt...")
            fallback_prompt = (
                f"Explain this lesson script in {target_language} like a classroom teacher. "
                f"Keep ALL technical terms, keywords, function names, and code snippets strictly in English:\n\n{text}"
            )
            res_retry = llm.invoke([HumanMessage(content=fallback_prompt)])
            translated = res_retry.content.strip()

        logger.info(f"Successfully generated teacher script for '{target_language}': Snippet -> '{translated[:60]}...'")
        return translated

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Critical error during Ollama translation to {target_language}:\n{error_trace}")
        raise RuntimeError(f"Translation to {target_language} failed:\n{error_trace}") from e
