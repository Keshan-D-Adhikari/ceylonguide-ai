import os
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

# LLM Configs
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GROQ_ROUTER_MODEL = os.getenv("GROQ_ROUTER_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_REASONING_MODEL = os.getenv("OPENROUTER_REASONING_MODEL", "deepseek/deepseek-r1")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "llama-3.3-70b-versatile")

# RAG Configs
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "5"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
