"""Application configuration utilities."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    """Runtime configuration loaded from environment variables."""

    api_title: str = "LLM Inference Optimization Demo"
    api_version: str = "1.0.0"
    model_path: str | None = os.getenv("MODEL_PATH")
    n_ctx: int = int(os.getenv("MODEL_CONTEXT_SIZE", "4096"))
    n_threads: int = int(os.getenv("MODEL_THREADS", str(os.cpu_count() or 4)))
    n_gpu_layers: int = int(os.getenv("MODEL_N_GPU_LAYERS", "0"))
    n_batch: int = int(os.getenv("MODEL_N_BATCH", "512"))
    max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))
    temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    use_mock: bool = os.getenv("LLAMA_USE_MOCK", "true").lower() in {"1", "true", "yes"}
    cors_origins: list[str] = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
