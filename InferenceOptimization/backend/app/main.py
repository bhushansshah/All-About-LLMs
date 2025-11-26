"""FastAPI application exposing the baseline LLM inference API."""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse

from .config import Settings, get_settings
from .metrics import MetricsTracker
from .models import GenerateRequest, MetricsSnapshot
from .services.llama_service import GenerationChunk, LlamaNotConfiguredError, LlamaService

_LOGGER = logging.getLogger(__name__)

app = FastAPI()

settings = get_settings()
metrics_tracker = MetricsTracker(window_seconds=60)
llama_service = LlamaService(settings=settings)

app.title = settings.api_title
app.version = settings.api_version

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_llama_service() -> LlamaService:
    """Provide an initialised LlamaService instance."""

    try:
        await llama_service.load()
    except LlamaNotConfiguredError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return llama_service


async def token_event_stream(
    generator: AsyncGenerator[GenerationChunk, None],
    started_at: float,
) -> AsyncGenerator[str, None]:
    """Translate token chunks into server-sent-event payloads."""

    token_total = 0
    try:
        async for chunk in generator:
            token_total += chunk.token_count
            payload = json.dumps({"type": "token", "content": chunk.text})
            yield f"data: {payload}\n\n"
    except Exception as exc:  # pragma: no cover - defensive path
        _LOGGER.exception("Generation failed: %%s", exc)
        error_payload = json.dumps({"type": "error", "message": str(exc)})
        yield f"data: {error_payload}\n\n"
        raise
    else:
        latency = metrics_tracker.finalize_request(started_at, token_total)
        metrics = metrics_tracker.snapshot()
        # Include per-request latency for the current call
        metrics_payload = {
            "type": "metrics",
            "metrics": {**metrics, "last_request_latency": latency, "last_request_tokens": token_total},
        }
        yield f"data: {json.dumps(metrics_payload)}\n\n"
        yield "data: [DONE]\n\n"


@app.on_event("startup")
async def on_startup() -> None:
    """Prime the llama service so requests are responsive."""

    try:
        await llama_service.load()
    except LlamaNotConfiguredError as exc:
        _LOGGER.warning("Backend started without a configured model: %s", exc)


@app.post(
    "/generate",
    response_class=StreamingResponse,
    summary="Stream a text completion from the baseline LLM",
)
async def generate(
    request: GenerateRequest,
    service: LlamaService = Depends(get_llama_service),
) -> StreamingResponse:
    """Generate tokens for the supplied prompt."""

    started_at = metrics_tracker.mark_start()
    generator = service.stream(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return StreamingResponse(
        token_event_stream(generator, started_at),
        media_type="text/event-stream",
    )


@app.get("/metrics", response_model=MetricsSnapshot)
async def read_metrics() -> MetricsSnapshot:
    """Return aggregate metrics over the sliding window."""

    snapshot = metrics_tracker.snapshot()
    return MetricsSnapshot(**snapshot)


@app.get("/health", response_class=PlainTextResponse)
async def healthcheck() -> str:
    """Simple health endpoint for readiness probes."""

    return "ok"
