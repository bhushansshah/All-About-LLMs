# LLM Inference Optimization Demo — Version 1 (Baseline)

This repository implements the baseline version of the **LLM Inference Optimization Demo**, showcasing end-to-end local inference with streaming token updates, latency tracking, and throughput metrics. It is the foundation that later versions (continuous batching, KV cache toggles, semantic cache) will build upon.

The project is split into a FastAPI backend and a React + Vite frontend, designed to be clean, modular, and easy to extend.

---

## Features Implemented in Version 1

- 🔌 **FastAPI backend** that forwards prompts directly to `llama.cpp` via `llama-cpp-python` (with an optional mock generator for quick demos).
- 🔄 **Server-Sent Events (SSE)** streaming for real-time token delivery to the frontend UI.
- 📊 **Metrics collection** covering tokens/sec, requests/sec, rolling average latency, and per-request totals.
- 🖥️ **React + Vite frontend** with an intuitive prompt form, live token stream viewer, and responsive metrics dashboard styled with Tailwind.
- ⚙️ **Configurable generation parameters** (temperature and max tokens) exposed through the UI.
- 🧱 **Extensible architecture** prepared for batching, caching, and semantic search enhancements in later versions.

---

## Project Structure

```
InferenceOptimization/
├─ README.md
├─ project_idea.md                     # Original project blueprint
├─ backend/
│  ├─ requirements.txt
│  ├─ .env.example
│  └─ app/
│     ├─ __init__.py
│     ├─ __main__.py
│     ├─ config.py                     # Environment-driven settings
│     ├─ main.py                       # FastAPI endpoints & streaming logic
│     ├─ metrics.py                    # Rolling metrics tracker
│     ├─ models.py                     # Pydantic request/response schemas
│     └─ services/
│        └─ llama_service.py           # llama.cpp + mock streaming service
└─ frontend/
   ├─ package.json
   ├─ tsconfig.json
   ├─ tsconfig.node.json
   ├─ vite.config.ts
   ├─ tailwind.config.js
   ├─ postcss.config.js
   ├─ index.html
   └─ src/
      ├─ App.tsx
      ├─ main.tsx
      ├─ index.css
      ├─ vite-env.d.ts
      ├─ hooks/
      │  └─ useLLMStream.ts            # Handles SSE + metrics polling
      └─ components/
         ├─ PromptForm.tsx             # Prompt input + generation controls
         ├─ TokenStream.tsx            # Live token stream viewer
         └─ MetricsPanel.tsx           # Throughput & latency dashboard
```

---

## Backend — FastAPI + llama.cpp

### Highlights
- `/generate` streams tokens via SSE, yielding events: `token`, `metrics`, `[DONE]`.
- `/metrics` reports a rolling one-minute snapshot for UI polling.
- `/health` returns `ok` for quick readiness checks.
- `LlamaService` lazily loads `llama.cpp` on first use and falls back to a mock generator when `LLAMA_USE_MOCK=true`.
- `MetricsTracker` maintains per-request latency and rolling averages using lightweight in-memory deques.

### Prerequisites
- Python **3.10+** (3.11 recommended)
- CMake build toolchain (required by `llama-cpp-python` when building from source)
- A GGML / GGUF model file (e.g., Qwen 4B or Llama 3 tier) for real inference

### Setup & Run

```bash
# 1. Create a virtual environment
cd backend
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
#   - Set MODEL_PATH to your GGML/GGUF file for real runs
#   - Leave LLAMA_USE_MOCK=true to explore the UI without a model

# 4. Launch the API (hot reload for development)
uvicorn app.main:app --reload --port 8000
```

### Switching to Real llama.cpp Inference
1. Install `llama-cpp-python` with Metal/CPU acceleration as needed (see [official docs](https://github.com/abetlen/llama-cpp-python)).
2. Update `.env`:
   ```ini
   MODEL_PATH=/absolute/path/to/model.gguf
   LLAMA_USE_MOCK=false
   MODEL_THREADS=<match your CPU cores>
   ```
3. Restart the backend. The startup logs will confirm when the model loads successfully.

> **Note:** Model loading can take several seconds. The backend pre-initialises the model on startup to keep requests responsive.

---

## Frontend — React + Vite + Tailwind

### Highlights
- Attractive gradient UI with responsive layout for desktop and tablet sizes.
- Prompt form includes sliders for temperature and max tokens, as well as clear/stop controls.
- Live token stream with automatic scroll and helpful placeholder messaging.
- Metrics dashboard highlights throughput and latency with subtle animations.
- Custom hook `useLLMStream` encapsulates SSE parsing, backpressure handling, AbortController usage, and metrics polling.

### Prerequisites
- Node.js **18+** (Node 20 recommended)
- npm (ships with Node) or Yarn/Pnpm if preferred

### Setup & Run

```bash
cd frontend
npm install
npm run dev
```

By default the frontend connects to `http://localhost:8000`. To point it elsewhere, create `frontend/.env.local` with:

```
VITE_API_BASE_URL=http://your-backend-host:port
```

The development server runs on `http://localhost:5173` (configurable via `VITE_PORT`).

---

## Using the Demo

1. Start the backend (`uvicorn app.main:app --reload`).
2. Start the frontend (`npm run dev`).
3. Visit `http://localhost:5173`.
4. Enter a prompt (e.g., “Explain how KV caching works”) and press **Generate**.
5. Watch tokens stream in real time, while metrics update both live (per request) and via polling.

The Stop button interrupts the current request via `AbortController`, allowing you to cancel long generations quickly.

---

## API Reference

| Endpoint     | Method | Description                                     |
|--------------|--------|-------------------------------------------------|
| `/generate`  | POST   | Streams token events for a given prompt (SSE).  |
| `/metrics`   | GET    | Returns rolling averages for UI dashboards.     |
| `/health`    | GET    | Basic readiness probe (`ok`).                   |

### `/generate` Request Body
```json
{
  "prompt": "string",
  "max_tokens": 256,
  "temperature": 0.7
}
```

### SSE Event Payloads
- `{"type": "token", "content": "partial text"}` — streamed content chunks
- `{"type": "metrics", "metrics": {...}}` — final metrics after completion
- `{"type": "error", "message": "..."}` — emitted if generation fails
- `[DONE]` — terminator event

---

## Extensibility Notes (Future Versions)

The codebase was structured with upcoming optimisations in mind:
- **Version 2** will hook additional modules into the service layer for KV cache toggles and async batching queues.
- **Version 3** can add a semantic cache service (FAISS + embeddings) alongside new endpoints and UI toggles.
- Metrics collection is centralised to make it easy to record new counters (cache hits, batch sizes, etc.).

---

## Troubleshooting

- **Model load errors:** Ensure `MODEL_PATH` points to a valid GGML/GGUF file and that `llama-cpp-python` has been installed with the right CPU/GPU flags.
- **SSE not streaming:** Check that your browser supports `ReadableStream` (all modern evergreen browsers do). Also confirm CORS settings (`CORS_ALLOWED_ORIGINS`).
- **High latency in mock mode:** The mock generator intentionally sleeps between tokens to mimic streaming; disable mock mode for real performance measurements.

---

## License

This project is provided under the MIT License. See `LICENSE` (add one if needed) for details.
