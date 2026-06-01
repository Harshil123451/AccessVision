# AccessVision Load Testing Guide

This directory contains a Locust-based load testing suite designed to stress-test and benchmark the AccessVision multimodal AI backend.

---

## 1. Directory Structure

```
load_tests/
├── test_images/                 # Preloaded target PNG image assets
│   ├── cars.png
│   ├── traffic.png
│   ├── room.png
│   ├── people.png
│   └── sign.png
├── reports/                     # Automatically generated CSV test reports
├── locustfile.py                # Core Locust User tasks and event hooks
├── run_load_test.ps1            # Automation script for scenarios
└── README_LOAD_TEST.md          # This instruction guide
```

---

## 2. Running the Load Tests

You can execute tests in both **interactive UI** and **headless CLI** modes.

### A. Automatic Headless Run via PowerShell
We provide standard scenario profiles (users and ramp-up rate) using the `run_load_test.ps1` script:

- **Sanity Profile** (5 concurrent users, 1 user/sec spawn rate):
  ```powershell
  powershell -File "load_tests/run_load_test.ps1" -Scenario sanity -Duration 1m
  ```
- **Standard Load Profile** (50 concurrent users, 5 users/sec spawn rate):
  ```powershell
  powershell -File "load_tests/run_load_test.ps1" -Scenario load -Duration 5m
  ```
- **Stress Profile** (200 concurrent users, 10 users/sec spawn rate):
  ```powershell
  powershell -File "load_tests/run_load_test.ps1" -Scenario stress -Duration 10m
  ```

### B. Interactive Web UI Mode
To launch Locust and open the web dashboard (`http://127.0.0.1:8089`):
```powershell
powershell -File "load_tests/run_load_test.ps1" -Headless $false
```

---

## 3. Interpretation Guide for Results

Locust generates several reports inside the `load_tests/reports/` directory. Open the CSV files or look at the console summary upon test completion to analyze the following metrics:

### A. Average & P95 Latency
- **Average Latency**: The arithmetic mean duration of successful requests. If it grows linearly with user count, the system is reaching saturation.
- **P95 Latency**: 95% of users experienced a latency lower than this value. This is the key metric for accessibility/real-time narration where predictable response times are critical.
- **Target Performance Benchmarks**:
  - `/api/v1/health` : `< 100ms` (IO-bound baseline)
  - `/api/v1/detect/objects` : `< 2.0s` (YOLO model execution)
  - `/api/v1/reason/query` : `< 4.0s` (Grounded router orchestration)
  - `/api/v1/scene/analyze` : `< 5.0s` (Concurrent YOLO + BLIP)

### B. Request Throughput (RPS)
- Represents how many requests per second the backend successfully completed.
- If RPS flattens out while concurrent users continue to rise, the system has hit a processing limit (bottleneck).

### C. Failure Rate
- **Target**: `< 1%` failure rate under standard load.
- Common failure causes under stress:
  - `HTTP 502 (Bad Gateway)` / `HTTP 504 (Gateway Timeout)`: Model inference task timed out.
  - `HTTP 422 (Unprocessable Entity)`: Schema verification failed (indicates corrupted image payload uploads or empty reasoning responses).
  - `HTTP 401 (Unauthorized)` / `HTTP 403 (Forbidden)`: Incorrect or missing `X-API-Key` headers.

---

## 4. Bottleneck Diagnosis Suggestions for AI Inference

When executing the **Stress Profile** (200+ users), look for these common bottlenecks:

### A. CPU / RAM Bottlenecks
- **Symptoms**: CPU cores pinned to 100%, high system RAM usage, request latency growing exponentially.
- **Reason**: Python's Global Interpreter Lock (GIL) serializes execution. While we use `run_in_thread` to run blocking inference in a background thread executor, heavy multi-threading can cause context-switching overhead and GIL thrashing.
- **Remedy**:
  - Increase the ASGI server workers. Run Uvicorn with multiple workers matching CPU physical cores:
    ```powershell
    uvicorn app.main:app --workers 4
    ```

### B. GPU VRAM Bottlenecks
- **Symptoms**: CUDA Out Of Memory (OOM) errors, uvicorn crashing, or sudden drop in GPU computing utilization (`htop`/`nvidia-smi` logs show VRAM at 100%).
- **Reason**: Model weights and batch buffers exceed available GPU memory.
- **Remedy**:
  - Keep models pinned to VRAM instead of constantly loading/unloading (Registry manages singletons).
  - Implement a request queue or limit concurrency to serialize GPU queries.

### C. Network Bandwidth Bottlenecks
- **Symptoms**: Client-side logs show slow upload times, but server-side model processing telemetry (`inference_ms`) is low.
- **Reason**: AccessVision uploads raw image payloads. Concurrent high-resolution uploads exhaust network bandwidth.
- **Remedy**:
  - Compress or downscale images on the client side (e.g. limit max dimension to 640px) before sending them to the backend.

---

## 5. Performance Monitoring Commands

Use these terminal commands during load tests to monitor system vitals:

### CPU and System RAM
- **Windows PowerShell**:
  ```powershell
  Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 -Property Name, CPU, WorkingSet
  ```
- **Linux / WSL**:
  ```bash
  htop
  ```

### GPU Utilization & VRAM (NVIDIA)
- Track GPU memory usage and temperature in real time (refreshed every 1 second):
  ```powershell
  nvidia-smi -l 1
  ```
