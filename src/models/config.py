import os
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    """
    Attempts to retrieve a config value/secret from:
    1. Streamlit secrets (if running in Streamlit environment)
    2. OS Environment Variables
    """
    try:
        import streamlit as st
        # Verify if st.secrets is available and key exists
        if hasattr(st, "secrets") and st.secrets is not None and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
        
    return os.getenv(key, default)

# LLM API Keys
GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY", "")

# Configured model names
GROQ_ROUTER_MODEL = get_secret("GROQ_ROUTER_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_REASONING_MODEL = get_secret("OPENROUTER_REASONING_MODEL", "deepseek/deepseek-r1")
CRITIC_MODEL = get_secret("CRITIC_MODEL", "llama-3.3-70b-versatile")

# RAG / Vector Store Configuration
CHROMA_DIR = get_secret("CHROMA_DIR", "data/chroma")
CHUNK_SIZE = int(get_secret("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(get_secret("CHUNK_OVERLAP", "120"))
TOP_K = int(get_secret("TOP_K", "5"))
