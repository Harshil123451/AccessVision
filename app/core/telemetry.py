import contextvars
import time
import os
import psutil
import torch
import logging
import json
import uuid
import asyncio
from contextlib import contextmanager
from typing import Dict, Any, List, Optional

logger = logging.getLogger("accessvision")

# Request-scoped context variable
request_telemetry_var: contextvars.ContextVar[Optional['RequestTelemetry']] = contextvars.ContextVar(
    "request_telemetry", default=None
)

class RequestTelemetry:
    """Stores execution timings, memory overhead, cache status, and queue states for an API request."""
    def __init__(self, request_id: str, endpoint: str):
        self.request_id = request_id
        self.endpoint = endpoint
        self.start_time = time.perf_counter()
        self.start_memory = self.get_ram_usage()
        self.start_gpu = self.get_gpu_usage()
        
        # Timing metrics (in ms)
        self.timings: Dict[str, float] = {}
        
        # Cache stats
        self.cache_hit: bool = False
        
        # Concurrency
        self.active_inference_tasks: int = 0
        
        # Trace messages
        self.traces: List[str] = []

    @staticmethod
    def get_ram_usage() -> float:
        """Returns resident set size (RSS) memory of the current process in MB."""
        try:
            process = psutil.Process(os.getpid())
            return round(process.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return 0.0

    @staticmethod
    def get_gpu_usage() -> float:
        """Returns allocated GPU memory in MB if CUDA is available."""
        try:
            if torch.cuda.is_available():
                return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass
        return 0.0

    def add_trace(self, message: str):
        self.traces.append(message)
        
    def record_timing(self, name: str, duration_ms: float):
        self.timings[name] = round(self.timings.get(name, 0.0) + duration_ms, 2)


class ObservableSemaphore:
    """Wraps asyncio.Semaphore to add detailed queue contention metrics and saturation monitoring."""
    def __init__(self, limit: int):
        self._sem = asyncio.Semaphore(limit)
        self._active_count = 0
        self._waiting_count = 0

    async def acquire(self) -> bool:
        self._waiting_count += 1
        start_time = time.perf_counter()
        
        telemetry = request_telemetry_var.get()
        if telemetry:
            telemetry.active_inference_tasks = self._active_count
            telemetry.add_trace(f"[QUEUE] active_inference_tasks={self._active_count}")
        
        try:
            await self._sem.acquire()
            self._active_count += 1
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            if telemetry:
                telemetry.record_timing("SEMAPHORE_WAIT", elapsed_ms)
                if elapsed_ms > 0.0:
                    telemetry.add_trace(f"[QUEUE] waited {elapsed_ms:.1f}ms for slot")
            
            if self._waiting_count > 1:
                logger.warning(
                    f"[QUEUE SATURATION] Queue buildup detected. "
                    f"Active tasks: {self._active_count}, Waiting tasks: {self._waiting_count - 1}"
                )
            return True
        finally:
            self._waiting_count -= 1

    def release(self):
        self._sem.release()
        self._active_count = max(0, self._active_count - 1)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def locked(self) -> bool:
        return self._sem.locked()

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def waiting_count(self) -> int:
        return self._waiting_count


@contextmanager
def trace_stage(stage_name: str, cache_status: Optional[str] = None):
    """Context manager to measure latency and resource allocation of individual execution stages."""
    telemetry = request_telemetry_var.get()
    if not telemetry:
        yield
        return
        
    start_time = time.perf_counter()
    start_ram = telemetry.get_ram_usage()
    start_gpu = telemetry.get_gpu_usage()
    
    try:
        yield
    finally:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        end_ram = telemetry.get_ram_usage()
        end_gpu = telemetry.get_gpu_usage()
        
        telemetry.record_timing(stage_name, elapsed_ms)
        
        # Track memory growth during this stage
        ram_growth = round(end_ram - start_ram, 2)
        gpu_growth = round(end_gpu - start_gpu, 2)
        
        # If there's memory growth, report it
        if ram_growth > 0.1:
            telemetry.add_trace(f"[MEM] {stage_name} RAM allocation spike: +{ram_growth}MB")
        if gpu_growth > 0.1:
            telemetry.add_trace(f"[MEM] {stage_name} GPU VRAM allocation spike: +{gpu_growth}MB")
            
        if cache_status:
            telemetry.add_trace(cache_status)
            
        telemetry.add_trace(f"[{stage_name}] {elapsed_ms:.1f}ms")


def get_current_telemetry() -> Optional[RequestTelemetry]:
    """Helper to fetch the current request's telemetry object."""
    return request_telemetry_var.get()


def log_console_trace(telemetry: RequestTelemetry):
    """Logs a clean, chronological trace of request events (similar to tail latency dumps)."""
    lines = [f"[REQ {telemetry.request_id}]"]
    for t in telemetry.traces:
        lines.append(f"  {t}")
    lines.append(f"  [TOTAL] {telemetry.timings.get('TOTAL', 0.0):.1f}ms")
    # Log as a single block to ensure standard stdout lines are grouped
    logger.info("\n" + "\n".join(lines))


def log_structured(telemetry: RequestTelemetry):
    """Generates structured JSON logs for backend monitoring integrations (Grafana Loki, ELK, Datadog)."""
    log_data = {
        "request_id": telemetry.request_id,
        "endpoint": telemetry.endpoint,
        "total_ms": telemetry.timings.get("TOTAL", 0.0),
        "preprocess_ms": telemetry.timings.get("PREPROCESS", 0.0),
        "decode_ms": telemetry.timings.get("IMAGE_DECODE", 0.0),
        "resize_ms": telemetry.timings.get("RESIZE", 0.0),
        "jpeg_optimize_ms": telemetry.timings.get("JPEG_OPTIMIZE", 0.0),
        "yolo_ms": telemetry.timings.get("YOLO", 0.0),
        "blip_ms": telemetry.timings.get("BLIP", 0.0),
        "vqa_ms": telemetry.timings.get("VQA", 0.0),
        "crop_ms": telemetry.timings.get("CROP", 0.0),
        "color_ms": telemetry.timings.get("COLOR", 0.0),
        "narration_ms": telemetry.timings.get("NARRATION", 0.0),
        "queue_wait_ms": telemetry.timings.get("SEMAPHORE_WAIT", 0.0),
        "cache_lookup_ms": telemetry.timings.get("CACHE_LOOKUP", 0.0),
        "cache_hit": telemetry.cache_hit,
        "ram_growth_mb": telemetry.timings.get("RAM_GROWTH_MB", 0.0),
        "gpu_growth_mb": telemetry.timings.get("GPU_GROWTH_MB", 0.0),
        "ram_usage_mb": telemetry.get_ram_usage(),
        "active_inference_tasks": telemetry.active_inference_tasks
    }
    logger.info(f"STRUCTURED_JSON: {json.dumps(log_data)}")


def check_slow_request(telemetry: RequestTelemetry):
    """Checks latencies against thresholds and logs standard alerting metrics."""
    total_ms = telemetry.timings.get("TOTAL", 0.0)
    yolo_ms = telemetry.timings.get("YOLO", 0.0)
    vqa_ms = telemetry.timings.get("VQA", 0.0)
    
    is_slow = False
    reasons = []
    if yolo_ms > 1000:
        is_slow = True
        reasons.append(f"yolo_latency={yolo_ms:.1f}ms")
    if vqa_ms > 3000:
        is_slow = True
        reasons.append(f"vqa_latency={vqa_ms:.1f}ms")
    if total_ms > 5000:
        is_slow = True
        reasons.append(f"total_latency={total_ms:.1f}ms")
        
    if is_slow:
        logger.warning(
            f"[SLOW REQUEST] request_id={telemetry.request_id} endpoint={telemetry.endpoint} "
            f"total_latency={total_ms:.1f}ms vqa_latency={vqa_ms:.1f}ms yolo_latency={yolo_ms:.1f}ms "
            f"reasons={', '.join(reasons)}"
        )
