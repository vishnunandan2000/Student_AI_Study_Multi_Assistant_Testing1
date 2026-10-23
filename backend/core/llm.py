import logging
import traceback
import urllib.request
import urllib.error
import json
from typing import Dict, Any
from backend.core.config import settings

# Configure module logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ChatOllama with fallback for version compatibility
try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        raise ImportError(
            "Neither 'langchain_ollama' nor 'langchain_community' is installed. "
            "Please install langchain-ollama using: pip install langchain-ollama"
        )


def check_ollama_connection() -> Dict[str, Any]:
    """
    Verifies that the local Ollama server is running at OLLAMA_BASE_URL
    and checks if MODEL_NAME is pulled and ready.

    Returns:
        Dict[str, Any]: Health status dictionary with keys:
            'connected' (bool), 'model_available' (bool), 'message' (str), 'models' (list)
    """
    base_url = settings.OLLAMA_BASE_URL.rstrip('/')
    target_model = settings.MODEL_NAME

    logger.info(f"Checking Ollama server connectivity at {base_url}...")

    try:
        # Request tags endpoint to inspect available models
        tags_url = f"{base_url}/api/tags"
        req = urllib.request.Request(tags_url, headers={'User-Agent': 'Python/StudentAssistant'})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                models_info = data.get('models', [])
                model_names = [m.get('name', '') for m in models_info]

                # Check if configured model or model prefix is loaded
                model_found = any(
                    target_model == name or target_model.split(':')[0] in name
                    for name in model_names
                )

                if model_found:
                    msg = f"Ollama online at {base_url}. Model '{target_model}' is ready."
                    logger.info(msg)
                    return {
                        'connected': True,
                        'model_available': True,
                        'message': msg,
                        'models': model_names
                    }
                else:
                    msg = (
                        f"Ollama online at {base_url}, but model '{target_model}' is not pulled yet. "
                        f"Please run: ollama pull {target_model}"
                    )
                    logger.warning(msg)
                    return {
                        'connected': True,
                        'model_available': False,
                        'message': msg,
                        'models': model_names
                    }

    except urllib.error.URLError as e:
        msg = (
            f"Unable to connect to local Ollama server at {base_url}. "
            f"Please ensure Ollama is installed and running (`ollama serve`). Error: {e.reason}"
        )
        logger.error(msg)
        return {'connected': False, 'model_available': False, 'message': msg, 'models': []}
    except Exception as e:
        msg = f"Unexpected error while probing Ollama server at {base_url}: {e}"
        logger.error(f"{msg}\n{traceback.format_exc()}")
        return {'connected': False, 'model_available': False, 'message': msg, 'models': []}


def get_llm() -> ChatOllama:
    """
    Factory function returning the centralized ChatOllama instance.
    Exclusively utilizes local llama3.2:latest across all project agents and workflows.

    Returns:
        ChatOllama: Production-ready local llama3.2:latest LLM instance.

    Raises:
        RuntimeError: If Ollama server is unreachable or fails instantiation.
    """
    target_model = "llama3.2:latest"
    logger.info(
        f"Instantiating local ChatOllama model '{target_model}' "
        f"at base URL '{settings.OLLAMA_BASE_URL}' (temperature: {settings.LLM_TEMPERATURE})"
    )

    try:
        llm = ChatOllama(
            model=target_model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            repeat_penalty=settings.LLM_REPEAT_PENALTY,
            num_ctx=settings.LLM_NUM_CTX,
            seed=settings.LLM_SEED,
            timeout=settings.LLM_TIMEOUT,
        )
        return llm
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Failed to instantiate ChatOllama:\n{error_trace}")
        raise RuntimeError(f"ChatOllama instantiation failed:\n{error_trace}") from e

