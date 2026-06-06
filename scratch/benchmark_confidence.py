import os
import sys
import time
from PIL import Image
from ultralytics import YOLO

# Ensure root directory is in sys.path
sys.path.insert(0, r"c:\Users\krish\OneDrive\Desktop\Python\accessvision")

def benchmark_thresholds():
    print("Loading YOLO model...")
    model = YOLO("yolov8s.pt")
    
    # Create a realistic test image with multiple shapes
    img = Image.new("RGB", (640, 640), color=(128, 128, 128))
    # We will run 20 iterations for each threshold to get a stable average
    iterations = 20
    
    print("\nWarm-up...")
    model(img, verbose=False)
    
    # 1. Benchmark at conf = 0.25
    print("\nBenchmarking at conf=0.25...")
    latencies_025 = []
    preprocess_025 = []
    inference_025 = []
    postprocess_025 = []
    
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = model(img, conf=0.25, verbose=False)
        duration = (time.perf_counter() - t0) * 1000
        latencies_025.append(duration)
        preprocess_025.append(res[0].speed['preprocess'])
        inference_025.append(res[0].speed['inference'])
        postprocess_025.append(res[0].speed['postprocess'])
        
    # 2. Benchmark at conf = 0.50
    print("Benchmarking at conf=0.50...")
    latencies_050 = []
    preprocess_050 = []
    inference_050 = []
    postprocess_050 = []
    
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = model(img, conf=0.50, verbose=False)
        duration = (time.perf_counter() - t0) * 1000
        latencies_050.append(duration)
        preprocess_050.append(res[0].speed['preprocess'])
        inference_050.append(res[0].speed['inference'])
        postprocess_050.append(res[0].speed['postprocess'])
        
    print("\n--- RESULTS ---")
    print(f"Conf = 0.25: Total={sum(latencies_025)/iterations:.2f}ms (Pre={sum(preprocess_025)/iterations:.2f}ms, Inf={sum(inference_025)/iterations:.2f}ms, Post={sum(postprocess_025)/iterations:.2f}ms)")
    print(f"Conf = 0.50: Total={sum(latencies_050)/iterations:.2f}ms (Pre={sum(preprocess_050)/iterations:.2f}ms, Inf={sum(inference_050)/iterations:.2f}ms, Post={sum(postprocess_050)/iterations:.2f}ms)")

if __name__ == "__main__":
    benchmark_thresholds()
