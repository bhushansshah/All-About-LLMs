"""Pydantic models for request and response payloads."""
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Schema for a text generation request."""

    prompt: str = Field(..., min_length=1, max_length=4096, description="User-provided prompt")
    max_tokens: int = Field(256, ge=1, le=1024, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    use_optimizations: bool = Field(False, description="Enable Version 2 optimizations (batching + KV cache)")


class MetricsSnapshot(BaseModel):
    """Aggregated metrics reported to the frontend."""

    tokens_per_sec: float = Field(0.0, description="Average tokens generated per second (rolling window)")
    requests_per_sec: float = Field(0.0, description="Average requests served per second (rolling window)")
    average_latency: float = Field(0.0, description="Average request latency in seconds (rolling window)")
    last_request_latency: float = Field(0.0, description="Latency for the most recent request")
    last_request_tokens: int = Field(0, description="Token count for the most recent request")
    average_batch_size: float = Field(0.0, description="Average batch size over the last window")
