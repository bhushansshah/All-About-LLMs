Project Title: **LLM Inference Optimization Demo (FastAPI + React + GGML)**

---

## **Project Overview**

This project demonstrates how different LLM inference optimizations affect latency, throughput, and user experience on a local system using **GGML / llama.cpp**, **FastAPI**, and a **React + Vite** frontend. The project is implemented in three versions:

### **Version 1 — Baseline (No Optimizations)**

* Users can query an LLM running on GGML / llama.cpp (CPU-based).
* No optimization mechanisms implemented.
* UI shows basic metrics: tokens/sec, requests/sec.
* Acts as the control baseline for comparison.

### **Version 2 — KV Cache + Simple Continuous Batching**

* Back-end introduces optimizations:

  * KV cache from llama.cpp (native).
  * A simple continuous batching mechanism (custom async batcher).
* Users can toggle optimizations.
* UI shows improved latency and throughput metrics.

### **Version 3 — Semantic Cache**

* Adds FAISS-based semantic caching.
* Before sending a generation request, the back-end checks if a similar query already exists.
* If a cache hit occurs, the result is returned instantly.
* UI reports cache hits, similarity scores, latency savings.

---

## **Technologies Used**

### **Backend (FastAPI)**

* Handles incoming requests (`/generate`).
* Manages async queue for batching.
* Interacts with llama.cpp through `llama-cpp-python`.
* Tracks metrics (latency, throughput, cache stats).

### **Frontend (React + Vite)**

* Clean and responsive UI.
* Interface for entering prompts.
* Optimization toggles (Version 2+).
* Displays token stream, tokens/sec, requests/sec, and cache metrics.

### **Model Runtime (GGML / llama.cpp)**

* Runs quantized Qwen3 4B mdodel or similar size llama model 
* KV cache directly supported.
* Lightweight and Apple CPU-friendly.

### **Semantic Cache (Version 3 only)**

* FAISS + small embedding model.
* Stores: prompt embedding → full generated text.

---

## **Architecture Overview**

### **1. Frontend (React + Vite)**

* Text input box for prompt.
* Button to send request.
* Toggle switches:

  * **Enable KV Cache + Batching** (v2)
  * **Enable Semantic Cache** (v3)
* Metrics panel:

  * Tokens/sec
  * Requests/sec
  * Latency
  * Cache hits (v3)
* Token-by-token streaming display.

### **2. Backend (FastAPI)**

#### **Version 1:** Simple Flow

* Request → llama.cpp → return result.
* No parallel request merging.
* Metrics collected:

  * Throughput: tokens/sec
  * Requests/sec
  * Latency

#### **Version 2:** Optimized Flow

* Add an async queue of incoming requests.
* Every X ms, the batcher takes available requests and executes them together.
* KV cache is used internally by llama.cpp for incremental generation.
* Metrics added:

  * Average batch size
  * KV cache reuse rate

#### **Version 3:** Semantic Cache

* On new request:

  1. Compute embedding.
  2. Query FAISS.
  3. If similar prompt exists (cosine similarity > threshold), return cached output.
  4. Otherwise, generate normally and store in cache.
* Additional metrics:

  * Cache hit rate
  * Cache latency savings

---

## **Detailed Version Breakdown**

---

## **Version 1 — Baseline System (No Optimizations)**

### Features

* Fully functional interface for querying the LLM.
* Backend simply forwards prompt to llama.cpp.
* No batching.
* No caching.
* Stream tokens back to frontend.

### Metrics to Display

* **Tokens/sec:** measure speed of model.
* **Requests/sec:** track throughput.
* **Latency per request:** round-trip measurement.

### Purpose

* Establish baseline for comparison with optimized versions.

---

## **Version 2 — Add KV Cache + Simple Continuous Batching**

### New Features

* Checkbox/Toggle in frontend: **Enable Optimizations**.
* Backend creates async queue and batcher:

  * Collects requests every 10–30ms.
  * Sends them in one batch to llama.cpp.

### KV Cache

* Provided automatically by llama.cpp backend.
* Reuses keys/values from previously generated tokens.
* Helps especially in long prompts or iterative interactions.

### Metrics to Display

* **Tokens/sec (optimized)**
* **Requests/sec (optimized)**
* **Average batch size**
* **Latency reduction (%)**
* **KV cache usage stats** (optional)

### Expected Outcomes

* Higher throughput.
* Lower per-request latency under concurrent load.
* Significantly improved performance vs version 1.

---

## **Version 3 — Semantic Cache**

### New Features

* Checkbox: **Enable Semantic Cache**.
* FAISS index stores prompt embeddings and corresponding LLM outputs.
* On new request:

  * If similar prompt exists → immediate return + no generation.

### Metrics to Display

* **Cache hit/miss counts**
* **Similarity scores**
* **Latency comparison: cached vs non-cached**

### Expected Outcomes

* Instant responses for repeated or similar queries.
* Large latency reduction for repetitive workloads.

---

## **Project Flow Summary**

1. **User enters prompt** → frontend sends request.
2. Backend checks for enabled optimizations.
3. **Version 1:** direct llama.cpp call.
4. **Version 2:** if enabled → queued + batched generation.
5. **Version 3:** check semantic cache → maybe skip generation.
6. Backend streams tokens back to frontend.
7. Metrics collected for each step.
8. Frontend updates UI with metrics.

---

## **Libraries & Tools**

### Backend

* `fastapi`
* `uvicorn` (server)
* `llama-cpp-python`
* `asyncio` (batching)
* `pydantic`

### Frontend

* `React + Vite`
* `Tailwind` for styling
* WebSockets for streaming tokens

### Semantic Cache (v3)

* `FAISS`
* `sentence-transformers` or `bge-small` embedding model

### Dev/Perf Tools

* `Locust` or custom asyncio load generator
* `Prometheus` (optional) for metric collection
* `Plotly` or `Matplotlib` for charts

---

## **Expected Outcomes Across Versions**

### Version 1

* Simple working LLM client.
* Baseline metrics logged.

### Version 2

* Higher throughput under concurrency.
* Faster average response times.
* Lower variability in latency.

### Version 3

* Dramatic latency reductions for repeated queries.
* Better overall system responsiveness.

---
