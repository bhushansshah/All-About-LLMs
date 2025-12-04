"""Abstractions for interacting with llama.cpp or a mock generator."""
from __future__ import annotations

import asyncio
import threading
import queue
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional
import logging
from collections import deque

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


@dataclass
class BatchRequest:
    """State for a request being processed."""
    request_id: int
    prompt: str
    max_tokens: int
    temperature: float
    output_queue: queue.Queue = field(default_factory=queue.Queue)


class LlamaService:
    """Service wrapper around llama.cpp with an optional mock implementation.
    
    Version 2 Optimizations:
    - Requests are queued and processed by a dedicated worker thread
    - This prevents blocking the main event loop
    - KV cache is utilized automatically by llama.cpp for each request
    - Throughput is improved under concurrent load by efficient queuing
    
    Note: True token-level interleaving (continuous batching) is not supported
    by llama-cpp-python's high-level API. For that, you would need to use
    vLLM or the low-level llama.cpp C API with manual KV cache slot management.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llama = None
        self._mode = "mock" if settings.use_mock else "llama"
        self._loaded = False
        self._batch_size_history: deque[int] = deque(maxlen=100)
        
        # For Version 2: Thread-safe request processing
        self._request_counter = 0
        self._counter_lock = threading.Lock()
        self._pending_requests: queue.Queue[BatchRequest] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Track concurrent requests for metrics
        self._active_request_count = 0
        self._count_lock = threading.Lock()

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
        except ImportError as exc:
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

        # Start the worker thread for optimized processing
        if self._mode == "llama" and self._worker_thread is None:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            _LOGGER.info("Request worker thread started.")

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        use_optimizations: bool = False,
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

        if use_optimizations:
            # Version 2: Submit to the worker queue
            # Benefits: Non-blocking, efficient queuing, metrics tracking
            with self._counter_lock:
                self._request_counter += 1
                request_id = self._request_counter

            request = BatchRequest(
                request_id=request_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            with self._count_lock:
                self._active_request_count += 1
                self._batch_size_history.append(self._active_request_count)
            
            self._pending_requests.put(request)
            _LOGGER.info(f"Request {request_id} submitted (queue size: {self._pending_requests.qsize()})")

            # Read from the request's output queue
            loop = asyncio.get_running_loop()
            try:
                while True:
                    event = await loop.run_in_executor(
                        None, 
                        lambda: request.output_queue.get(timeout=120.0)
                    )
                    
                    if event.kind == "token" and event.payload:
                        yield GenerationChunk(
                            text=event.payload,
                            token_count=self._estimate_token_count(event.payload),
                        )
                    elif event.kind == "error" and event.error:
                        raise event.error
                    elif event.kind == "end":
                        break
            except queue.Empty:
                _LOGGER.error(f"Request {request_id} timed out")
                raise RuntimeError("Generation timed out")
            finally:
                with self._count_lock:
                    self._active_request_count = max(0, self._active_request_count - 1)
            return

        # Version 1: Direct execution (Baseline) - blocks on model
        loop = asyncio.get_running_loop()
        async_queue: asyncio.Queue[_QueueEvent] = asyncio.Queue()

        def producer() -> None:
            try:
                template = self._format_prompt(prompt)
                iterator = self._llama.create_completion(
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
                        async_queue.put_nowait, _QueueEvent(kind="token", payload=text)
                    )
            except BaseException as exc:
                loop.call_soon_threadsafe(
                    async_queue.put_nowait, _QueueEvent(kind="error", error=exc)
                )
            finally:
                loop.call_soon_threadsafe(async_queue.put_nowait, _QueueEvent(kind="end"))

        producer_task = asyncio.create_task(asyncio.to_thread(producer))
        try:
            while True:
                event = await async_queue.get()
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

    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt with the chat template."""
        return (
            "<|im_start|>system\n"
            "You are a concise assistant. Answer factually.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _worker_loop(self) -> None:
        """
        Worker thread that processes requests sequentially from the queue.
        
        This approach:
        - Utilizes KV cache efficiently (llama.cpp handles this internally)
        - Prevents request starvation via FIFO queue
        - Streams tokens back to clients as they're generated
        - Doesn't block the async event loop
        
        Note: Requests are processed one at a time to avoid KV cache corruption.
        For true concurrent batch processing, use vLLM or similar frameworks.
        """
        _LOGGER.info("Worker loop starting...")
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for a request (with timeout to allow shutdown)
                try:
                    request = self._pending_requests.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                _LOGGER.info(f"Processing request {request.request_id}")
                
                # Process the request
                self._process_request(request)
                
            except Exception as e:
                _LOGGER.error(f"Worker loop error: {e}")
        
        _LOGGER.info("Worker loop shutting down...")

    def _process_request(self, request: BatchRequest) -> None:
        """Process a single request and stream tokens to its output queue."""
        try:
            template = self._format_prompt(request.prompt)
            
            iterator = self._llama.create_completion(
                prompt=template,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            )
            
            for chunk in iterator:
                text = chunk["choices"][0]["text"]
                if text:
                    request.output_queue.put(_QueueEvent(kind="token", payload=text))
            
            request.output_queue.put(_QueueEvent(kind="end"))
            _LOGGER.info(f"Request {request.request_id} completed")
            
        except Exception as e:
            _LOGGER.error(f"Error processing request {request.request_id}: {e}")
            request.output_queue.put(_QueueEvent(kind="error", error=e))

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

    @property
    def average_batch_size(self) -> float:
        """Return the average number of concurrent requests over recent history."""
        if not self._batch_size_history:
            return 0.0
        return sum(self._batch_size_history) / len(self._batch_size_history)

    @staticmethod   
    def _estimate_token_count(text: str) -> int:
        """Very rough heuristic for token counting."""

        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, len(stripped.split()))
