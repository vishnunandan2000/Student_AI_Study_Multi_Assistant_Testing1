"""
GenAI Instagram Intelligence Studio - Ollama Diagnostic Script
Verifies Ollama server connectivity and llama3.2:latest model availability.
"""

import sys
import os
import requests
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL

def check_ollama():
    print("=" * 60)
    print("Ollama Service Diagnostics")
    print("=" * 60)
    print(f"Target URL: {OLLAMA_BASE_URL}")
    print(f"Required Model: {OLLAMA_MODEL}")
    
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            print(f"Server Status: ONLINE (HTTP {resp.status_code})")
            print(f"Available Models: {models}")
            
            # Check for required model
            matches = [m for m in models if OLLAMA_MODEL in m]
            if matches:
                print(f"Model Check: [SUCCESS] Found matching model: {matches[0]}")
            else:
                print(f"Model Check: [WARNING] Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
        else:
            print(f"Server Status: HTTP {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Server Status: OFFLINE / UNREACHABLE ({e})")
        print("Tip: Ensure Ollama is running (`ollama serve`)")
    print("=" * 60)

if __name__ == "__main__":
    check_ollama()
