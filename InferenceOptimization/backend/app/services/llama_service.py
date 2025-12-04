"""Abstractions for interacting with llama.cpp or a mock generator."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Optional
import logging

from ..config import Settings

_LOGGER = logging.getLogger(__name__)


class LlamaNotConfiguredError(RuntimeError):
    """Raised when the llama backend is not available."""


@dataclass
class GenerationChunk:
    """A single streamed chunk of generated text."""

    text: str
    token_count: int


@dataclass
class _QueueEvent:
    kind: str
    payload: Optional[str] = None
    error: Optional[BaseException] = None


class LlamaService:
    """Service wrapper around llama.cpp with an optional mock implementation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llama = None
        self._mode = "mock" if settings.use_mock else "llama"
        self._loaded = False

    async def load(self) -> None:
        """Initialise the llama backend."""

        if self._loaded:
            return

        if self._mode == "mock":
            _LOGGER.warning("Starting in mock generation mode. Set LLAMA_USE_MOCK=false for real model inference.")
            self._loaded = True
            return

        if not self._settings.model_path:
            raise LlamaNotConfiguredError(
                "MODEL_PATH is not configured and mock mode is disabled."
            )

        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover - informative error path
            raise LlamaNotConfiguredError(
                "llama-cpp-python is not installed. Install it or enable mock mode."
            ) from exc

        _LOGGER.info("Loading llama.cpp model from %s", self._settings.model_path)
        self._llama = await asyncio.to_thread(
            Llama,
            model_path=self._settings.model_path,
            n_ctx=self._settings.n_ctx,
            n_threads=self._settings.n_threads,
            n_gpu_layers=self._settings.n_gpu_layers,
            logits_all=False,
            vocab_only=False,
            use_mmap=True,
            use_mlock=False,
        )
        self._loaded = True
        _LOGGER.info("Model loaded successfully.")

    async def stream(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> AsyncGenerator[GenerationChunk, None]:
        """Stream generated tokens for the provided prompt."""

        if not self._loaded:
            await self.load()

        if self._mode == "mock":
            async for chunk in self._mock_stream(prompt, max_tokens):
                yield chunk
            return

        if self._llama is None:
            raise LlamaNotConfiguredError("The llama model failed to load.")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[_QueueEvent] = asyncio.Queue()

        def producer() -> None:
            try:
                template = (
                    "<|im_start|>system\n"
                    "You are a concise assistant. Answer factually.<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"{prompt}\n"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                iterator = self._llama.create_completion(  # type: ignore[attr-defined]
                    prompt=template,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                for chunk in iterator:
                    text = chunk["choices"][0]["text"]
                    if not text:
                        continue
                    loop.call_soon_threadsafe(
                        queue.put_nowait, _QueueEvent(kind="token", payload=text)
                    )
            except BaseException as exc:  # pragma: no cover - defensive path
                loop.call_soon_threadsafe(
                    queue.put_nowait, _QueueEvent(kind="error", error=exc)
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _QueueEvent(kind="end"))

        producer_task = asyncio.create_task(asyncio.to_thread(producer))
        try:
            while True:
                event = await queue.get()
                if event.kind == "token" and event.payload:
                    yield GenerationChunk(
                        text=event.payload,
                        token_count=self._estimate_token_count(event.payload),
                    )
                elif event.kind == "error" and event.error:
                    raise event.error
                elif event.kind == "end":
                    break
        finally:
            await producer_task

    async def _mock_stream(
        self, prompt: str, max_tokens: int
    ) -> AsyncGenerator[GenerationChunk, None]:
        """Emit a deterministic mock completion for local development."""

        base_text = (
            "This is a mock response because no GGML model is configured. "
            "Update the backend configuration to stream real tokens."
        )
        seed_text = f"Prompt received: {prompt.strip()}\n\n{base_text}"
        words = seed_text.split()
        emitted = 0
        for word in words[:max_tokens]:
            await asyncio.sleep(0.1)
            yield GenerationChunk(text=word + " ", token_count=1)
            emitted += 1
        if emitted < max_tokens:
            await asyncio.sleep(0.05)
            yield GenerationChunk(text="[End of mock output]", token_count=4)

    @staticmethod   
    def _estimate_token_count(text: str) -> int:
        """Very rough heuristic for token counting."""

        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, len(stripped.split()))
