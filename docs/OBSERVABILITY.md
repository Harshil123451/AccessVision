# AccessVision — Telemetry, Observability & Concurrency Control

This document provides a deep technical breakdown of the logging, request tracing, resource management, and load-testing structures engineered into the AccessVision backend.

---

## 🔍 Request Correlation Tracing

To monitor backend pipelines under concurrent load, AccessVision uses a context-local tracing system. Because Python’s FastAPI server handles requests asynchronously, simple global logging variables would mix concurrent logs together. We resolve this by using Python's standard `contextvars` library to store a unique correlation ID per request.

### Implementation Details (`app/core/telemetry.py`)
Each incoming HTTP request generates a random, short alphanumeric correlation ID (e.g. `req_7f8a9`). This ID is stored in a context variable and injected into every log statement produced during the request's lifecycle.

Example log output under heavy concurrent load:
```text
[INFO] 2026-06-01 11:50:00 [req_7f8a9] Started POST /api/v1/reason/query
[INFO] 2026-06-01 11:50:00 [req_a2b3c] Started POST /api/v1/reason/query
[INFO] 2026-06-01 11:50:00 [req_7f8a9] [PREPROCESS] Image compressed from 3.2MB to 142KB
[INFO] 2026-06-01 11:50:01 [req_a2b3c] [PREPROCESS] Image compressed from 4.1MB to 160KB
[INFO] 2026-06-01 11:50:01 [req_7f8a9] [ROUTER] Route selected: Grounded Object Reasoning (Color)
[INFO] 2026-06-01 11:50:01 [req_a2b3c] [ROUTER] Route selected: Scene Narration (Fallback)
[INFO] 2026-06-01 11:50:02 [req_7f8a9] [YOLO] Identified 'backpack' at [120, 200, 340, 500] (conf: 0.91)
[INFO] 2026-06-01 11:50:02 [req_a2b3c] [BLIP] Generated scene caption: "a laptop sitting on a wooden desk"
[INFO] 2026-06-01 11:50:02 [req_7f8a9] [VQA] Executed BLIP VQA on cropped 'backpack' region (18ms)
[INFO] 2026-06-01 11:50:02 [req_a2b3c] Finished POST - Status: 200 OK (2012ms)
[INFO] 2026-06-01 11:50:03 [req_7f8a9] Finished POST - Status: 200 OK (2421ms)
```

Through these correlation logs, engineers can isolate the trace of a single user's request even when dozens of users are querying the server simultaneously.

---

## ⚙️ Concurrency Throttling & Resource Control

Deep learning models (YOLOv8, BLIP, VQA pipelines) are CPU/GPU-intensive. Allowing unlimited concurrent requests to query the models will quickly exhaust server VRAM or host memory, leading to Out-Of-Memory (OOM) crashes.

To safeguard system stability, AccessVision implements two primary resource defenses:

### 1. Concurrency Semaphores
We wrap the model inference sections in an asynchronous semaphore (`asyncio.Semaphore(limit)`).
- **Wait Queue**: If requests exceed the semaphore limit (configured via `SEMAPHORE_LIMIT` in `.env`), they wait in a non-blocking queue rather than invoking the GPU immediately.
- **Fair Scheduling**: Requests are processed in a first-in, first-out sequence as soon as a slot is freed by another request finishing its forward pass.

### 2. Async Thread Pools
Since standard deep learning inferences (using PyTorch/Ultralytics) run synchronous C++ code under the hood, calling them directly would freeze the FastAPI single-threaded event loop. We use `asyncio.to_thread` to delegate all model forward passes to a secondary worker thread pool, keeping the main thread free to handle new incoming HTTP request connections.

---

## 📈 Locust Load Testing

We validated backend stability and measured performance bottlenecks using Locust, an open-source load-testing tool.

### Testing Parameters
- **Simulated Users**: 50 concurrent active users.
- **Behavior**: Users continuously upload compressed camera frames and request scene narrations or ask questions.
- **Test Duration**: 10 minutes.

### Key Metrics Tracked
- **Queue Wait Time**: Telemetry measures how long a request waits in the semaphore queue before accessing the GPU.
- **Model Execution Duration**: Separate tracking for preprocessing, object detection (YOLO), and visual text generation (BLIP).
- **RAM/VRAM Profiling**: Monitored host memory usage to confirm the effectiveness of Python's garbage collection and PyTorch memory clearing.
