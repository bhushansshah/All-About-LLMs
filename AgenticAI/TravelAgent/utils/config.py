# utils/config.py
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)

def require_env(key: str) -> str:
    val = get_env(key)
    if not val:
        raise EnvironmentError(f"Environment variable {key} is required.")
    return val

def set_google_api_key(key: str) -> None:
    """
    Set the GOOGLE_API_KEY in the process environment.
    For production, use secret manager instead of writing to env.
    """
    if not key:
        raise ValueError("Google API key must be provided.")
    os.environ["GOOGLE_API_KEY"] = key
