import requests
from src.models import config

class APIKeyError(ValueError):
    """Exception raised when a required API key is missing or empty."""
    pass

class LLMProviderError(RuntimeError):
    """Exception raised when an API provider returns an error."""
    pass

def get_groq_client():
    """
    Returns an initialized Groq client.
    Raises APIKeyError if GROQ_API_KEY is not configured.
    """
    api_key = config.GROQ_API_KEY
    if not api_key:
        raise APIKeyError(
            "GROQ_API_KEY is missing. Please set it in your .env file or Streamlit secrets."
        )
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        raise ImportError(
            "groq package is not installed. Please run 'pip install groq'."
        )

def generate_groq_completion(
    messages: list[dict], 
    model: str = None, 
    temperature: float = 0.1,
    response_format: dict = None
) -> str:
    """
    Generates a chat completion using Groq.
    supports JSON mode by passing response_format={"type": "json_object"}.
    """
    client = get_groq_client()
    target_model = model or config.GROQ_ROUTER_MODEL
    
    try:
        kwargs = {
            "messages": messages,
            "model": target_model,
            "temperature": temperature
        }
        if response_format:
            kwargs["response_format"] = response_format
            
        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
    except Exception as e:
        raise LLMProviderError(f"Groq API error: {e}")

def generate_openrouter_completion(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.3
) -> str:
    """
    Generates a chat completion using OpenRouter.
    Uses requests to invoke OpenRouter completions endpoint.
    """
    api_key = config.OPENROUTER_API_KEY
    if not api_key:
        raise APIKeyError(
            "OPENROUTER_API_KEY is missing. Please set it in your .env file or Streamlit secrets."
        )
        
    target_model = model or config.OPENROUTER_REASONING_MODEL
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ceylonguide-ai", # Required by OpenRouter
        "X-Title": "CeylonGuide AI Travel Planner"
    }
    
    payload = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            # Avoid leaking raw authorization tokens in error messages
            raise LLMProviderError(
                f"OpenRouter returned status code {response.status_code}: {response.text[:200]}"
            )
            
        res_json = response.json()
        if "choices" not in res_json or not res_json["choices"]:
            raise LLMProviderError(f"Malformed OpenRouter response: {res_json}")
            
        return res_json["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise LLMProviderError(f"Network error calling OpenRouter: {e}")
    except Exception as e:
        raise LLMProviderError(f"OpenRouter error: {e}")
