import os
import sys
import asyncio
import time
from PIL import Image

# Ensure root directory is in sys.path
sys.path.insert(0, r"c:\Users\krish\OneDrive\Desktop\Python\accessvision")

from app.core.config import settings
from app.ai.registry import ModelRegistry
from app.services.detect_service import DetectService
from app.services.florence_service import FlorenceService
from app.utils.image import save_image_to_bytes

async def run_yolo_task(detect_service, img_bytes, task_id):
    t0 = time.perf_counter()
    # Disable cache to force actual inference
    DetectService._cache.clear()
    res = await detect_service.detect_objects(img_bytes)
    duration = (time.perf_counter() - t0) * 1000
    print(f"YOLO Task {task_id} completed in {duration:.1f}ms (Inference: {res.metrics.inference_ms:.1f}ms)")
    return duration

async def run_florence_task(florence_service, img_bytes, task_id):
    t0 = time.perf_counter()
    res = await florence_service.get_detailed_caption(img_bytes)
    duration = (time.perf_counter() - t0) * 1000
    print(f"Florence Task {task_id} completed in {duration:.1f}ms | Caption: '{res}'")
    return duration

async def simulate_concurrency():
    print("=================================================================")
    print("STARTING ACCESSVISION CONCURRENCY BENCHMARK")
    print("=================================================================")
    
    # 1. Warm up and initialize
    print("Initializing Model Registry & Pre-loading Weights (including warmup passes)...")
    t_load_start = time.perf_counter()
    ModelRegistry.load_all()
    print(f"Loading and Warmup completed in {(time.perf_counter() - t_load_start):.2f}s")
    
    detect_service = DetectService()
    florence_service = FlorenceService()
    
    # Create dummy images
    img = Image.new("RGB", (640, 640), color=(128, 128, 128))
    img_bytes = save_image_to_bytes(img)
    
    print("\n--- Running Single YOLO pass (Steady-state check) ---")
    await run_yolo_task(detect_service, img_bytes, 0)
    
    print("\n--- Running Single Florence-2 pass (Steady-state check) ---")
    await run_florence_task(florence_service, img_bytes, 0)
    
    print("\n--- Launching Concurrent Load Simulation ---")
    print("Spawning 5 parallel YOLO inferences and 2 parallel Florence caption tasks...")
    
    tasks = []
    # Spawn 5 YOLO requests
    for i in range(1, 6):
        tasks.append(run_yolo_task(detect_service, img_bytes, i))
    # Spawn 2 Florence requests
    for i in range(1, 3):
        tasks.append(run_florence_task(florence_service, img_bytes, i))
        
    t_concur_start = time.perf_counter()
    latencies = await asyncio.gather(*tasks)
    total_concur_time = (time.perf_counter() - t_concur_start) * 1000
    
    yolo_latencies = latencies[:5]
    florence_latencies = latencies[5:]
    
    print("\n=================================================================")
    print("CONCURRENCY RESULTS")
    print("=================================================================")
    print(f"Total concurrency simulation wall-clock time: {total_concur_time:.1f}ms")
    print(f"YOLO Latencies: min={min(yolo_latencies):.1f}ms, max={max(yolo_latencies):.1f}ms, avg={sum(yolo_latencies)/len(yolo_latencies):.1f}ms")
    print(f"Florence Latencies: min={min(florence_latencies):.1f}ms, max={max(florence_latencies):.1f}ms, avg={sum(florence_latencies)/len(florence_latencies):.1f}ms")
    print("=================================================================")

if __name__ == "__main__":
    asyncio.run(simulate_concurrency())
