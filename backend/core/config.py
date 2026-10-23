import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Local Ollama LLM Centralized Configuration (Exclusive: llama3.2:latest)
    MODEL_NAME: str = "llama3.2:latest"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
    LLM_REPEAT_PENALTY: float = float(os.getenv("LLM_REPEAT_PENALTY", "1.1"))
    LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "8192"))
    LLM_SEED: int = int(os.getenv("LLM_SEED", "42"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "300"))

    # GenAI Video Intelligence Studio Settings
    OLLAMA_MODEL: str = "llama3.2:latest"
    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
    MAX_VIDEO_DURATION_MINUTES: int = int(os.getenv("MAX_VIDEO_DURATION_MINUTES", "120"))
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
    DEFAULT_OUTPUT_LANGUAGE: str = os.getenv("DEFAULT_OUTPUT_LANGUAGE", "en")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

settings = Settings()

# Direct module-level accessors
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL
MODEL_NAME = settings.MODEL_NAME
MAX_UPLOAD_SIZE_MB = settings.MAX_UPLOAD_SIZE_MB
MAX_VIDEO_DURATION_MINUTES = settings.MAX_VIDEO_DURATION_MINUTES
MAX_CONCURRENT_JOBS = settings.MAX_CONCURRENT_JOBS
DEFAULT_OUTPUT_LANGUAGE = settings.DEFAULT_OUTPUT_LANGUAGE

