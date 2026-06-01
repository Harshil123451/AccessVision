import os
import random
import logging
from locust import HttpUser, task, between, events

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("accessvision-loadtest")

# Target API Key header (bypass or key check in backend depend on config)
HEADERS = {
    "X-API-Key": "development-secret-key"
}

# Image-to-Query Mapping to simulate realistic accessibility requests
MAPPING = [
    {
        "filename": "cars.png",
        "questions": ["What color is the car?", "How many cars are there?"]
    },
    {
        "filename": "traffic.png",
        "questions": ["Is there a traffic light?", "Read the sign."]
    },
    {
        "filename": "room.png",
        "questions": ["Describe the scene.", "Is there a chair nearby?"]
    },
    {
        "filename": "people.png",
        "questions": ["How many people are there?", "Who is in the room?"]
    },
    {
        "filename": "sign.png",
        "questions": ["Read the sign.", "What does the sign say?"]
    }
]

# In-memory image cache to avoid disk I/O bottlenecks during load testing
IMAGE_CACHE = {}

@events.test_start.add_listener
def preload_images(environment, **kwargs):
    """Loads all test images into memory before the load test begins."""
    logger.info("Starting load test. Preloading image assets into memory...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, "test_images")
    
    for item in MAPPING:
        name = item["filename"]
        path = os.path.join(images_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                IMAGE_CACHE[name] = f.read()
            logger.info(f"Loaded asset: {name} ({len(IMAGE_CACHE[name])} bytes)")
        else:
            logger.error(f"Required test image not found: {path}")

class AccessVisionUser(HttpUser):
    """Simulates a visually impaired user navigating with the AccessVision platform."""
    
    # Simulate real-world pause times (1.0 to 3.0 seconds) between actions
    wait_time = between(1.0, 3.0)

    @task(1)
    def health_check(self):
        """Tests the system health and baseline latency."""
        with self.client.get(
            "/api/v1/health", 
            headers=HEADERS, 
            name="Health Check", 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status code {response.status_code}")

    @task(2)
    def generate_caption(self):
        """Tests the image captioning model pipeline."""
        filename, image_bytes = self._get_random_image()
        if not image_bytes:
            return
        
        files = {
            "file": (filename, image_bytes, "image/png")
        }
        with self.client.post(
            "/api/v1/caption/generate", 
            files=files, 
            headers=HEADERS, 
            name="Caption Generation (BLIP)", 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Caption generate failed: {response.text}")

    @task(3)
    def detect_objects(self):
        """Tests object detection pipeline."""
        filename, image_bytes = self._get_random_image()
        if not image_bytes:
            return
        
        files = {
            "file": (filename, image_bytes, "image/png")
        }
        with self.client.post(
            "/api/v1/detect/objects", 
            files=files, 
            headers=HEADERS, 
            name="Object Detection (YOLO)", 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Detect objects failed: {response.text}")

    @task(2)
    def ask_vqa(self):
        """Tests visual question answering fallback pipeline."""
        filename, image_bytes = self._get_random_image()
        if not image_bytes:
            return
        
        question = self._get_random_question(filename)
        files = {
            "file": (filename, image_bytes, "image/png")
        }
        data = {
            "question": question
        }
        with self.client.post(
            "/api/v1/vqa/ask", 
            files=files, 
            data=data, 
            headers=HEADERS, 
            name="Visual Question Answering (VQA)", 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"VQA failed: {response.text}")

    @task(8)  # Heavily weighted task representing the primary user interaction
    def grounded_query(self):
        """Stress-tests the Question Router / Grounded Reasoning pipeline."""
        filename, image_bytes = self._get_random_image()
        if not image_bytes:
            return
        
        question = self._get_random_question(filename)
        files = {
            "file": (filename, image_bytes, "image/png")
        }
        data = {
            "question": question
        }
        
        with self.client.post(
            "/api/v1/reason/query", 
            files=files, 
            data=data, 
            headers=HEADERS, 
            name="Grounded Query Routing (Orchestrator)", 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    res_json = response.json()
                    # Validate schema items to monitor logic failures/hallucinations
                    if not res_json.get("success"):
                        response.failure("Logic failed: reasoning success field is False")
                    elif not res_json.get("answer"):
                        response.failure("Logic failed: answer field is empty")
                    else:
                        response.success()
                except Exception as e:
                    response.failure(f"JSON parsing error: {str(e)}")
            else:
                response.failure(f"Reasoning query failed: {response.text}")

    @task(4)
    def analyze_scene(self):
        """Stress-tests parallel YOLO/BLIP scene narration generation."""
        filename, image_bytes = self._get_random_image()
        if not image_bytes:
            return
        
        files = {
            "file": (filename, image_bytes, "image/png")
        }
        with self.client.post(
            "/api/v1/scene/analyze", 
            files=files, 
            headers=HEADERS, 
            name="Scene Analysis (YOLO + BLIP)", 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    res_json = response.json()
                    if not res_json.get("narration"):
                        response.failure("Scene Service narration field is empty")
                    else:
                        response.success()
                except Exception as e:
                    response.failure(f"JSON parsing error: {str(e)}")
            else:
                response.failure(f"Scene analyze failed: {response.text}")

    def _get_random_image(self):
        """Helper to get a random preloaded image filename and byte payload."""
        if not IMAGE_CACHE:
            return None, None
        filename = random.choice(list(IMAGE_CACHE.keys()))
        return filename, IMAGE_CACHE[filename]

    def _get_random_question(self, filename):
        """Helper to get a relevant query matching the selected image context."""
        for item in MAPPING:
            if item["filename"] == filename:
                return random.choice(item["questions"])
        return "Describe what you see."

# Thread-safe in-memory collections for post-run telemetry reporting
import time
from collections import defaultdict
import numpy as np
import csv

LATENCIES = defaultdict(list)
SLOW_REQUESTS = []

# Custom Event Listeners for performance telemetry & threshold logging
@events.request.add_listener
def log_performance_telemetry(request_type, name, response_time, response_length, response, exception, **kwargs):
    """Tracks latency outliers, extracts correlated request IDs, and records execution statistics."""
    request_id = "N/A"
    total_duration_ms = "N/A"
    
    if response and hasattr(response, "headers"):
        request_id = response.headers.get("X-Request-ID", "N/A")
        total_duration_ms = response.headers.get("X-Total-Duration-MS", "N/A")
        
    LATENCIES[name].append(response_time)

    # Thresholds in milliseconds
    THRESHOLDS = {
        "Health Check": 100,
        "Object Detection (YOLO)": 1000,
        "Visual Question Answering (VQA)": 3000,
        "Grounded Query Routing (Orchestrator)": 5000,
        "Scene Analysis (YOLO + BLIP)": 5000,
        "Caption Generation (BLIP)": 3000
    }
    
    # Check if the request exceeded its defined target threshold
    limit = THRESHOLDS.get(name, 5000)
    if response_time > limit:
        logger.warning(
            f"[SLOW LOCUST REQUEST] ID={request_id} | Endpoint='{name}' took {response_time:.1f}ms "
            f"(Threshold: {limit}ms). Server-side duration: {total_duration_ms}ms. Length: {response_length} bytes."
        )
        SLOW_REQUESTS.append((time.time(), request_id, name, response_time))
            
    # Track exceptions/timeouts specifically
    if exception:
        logger.error(f"REQUEST EXCEPTION: ID={request_id} | '{name}' failed with: {str(exception)}")

@events.test_stop.add_listener
def export_performance_metrics(environment, **kwargs):
    """Fires when load test finishes. Compiles p50, p95, p99, and worst-case, then saves statistics to CSV."""
    logger.info("Test run finished. Generating performance audits...")
    
    report_lines = [
        "",
        "==========================================================================================",
        "                             LATENCY PERCENTILES SUMMARY                                  ",
        "==========================================================================================",
        f"{'Request Pathway / Tag':<45} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10} | {'Worst (ms)':<10} | {'Count':<6}",
        "------------------------------------------------------------------------------------------"
    ]
    
    csv_rows = [["Endpoint/Tag", "RequestCount", "p50_ms", "p95_ms", "p99_ms", "WorstCase_ms"]]
    
    for endpoint, times in LATENCIES.items():
        if not times:
            continue
        times_arr = np.array(times)
        p50 = np.percentile(times_arr, 50)
        p95 = np.percentile(times_arr, 95)
        p99 = np.percentile(times_arr, 99)
        worst = np.max(times_arr)
        count = len(times)
        
        report_lines.append(
            f"{endpoint:<45} | {p50:<10.1f} | {p95:<10.1f} | {p99:<10.1f} | {worst:<10.1f} | {count:<6}"
        )
        csv_rows.append([endpoint, count, round(p50, 2), round(p95, 2), round(p99, 2), round(worst, 2)])
        
    report_lines.append("==========================================================================================")
    logger.info("\n".join(report_lines))
    
    # Export metrics
    os.makedirs("reports", exist_ok=True)
    metrics_filepath = "reports/latency_percentiles.csv"
    try:
        with open(metrics_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        logger.info(f"Successfully exported percentile metrics to: {os.path.abspath(metrics_filepath)}")
    except Exception as e:
        logger.error(f"Failed to export CSV percentiles: {str(e)}")
        
    # Export slow requests
    if SLOW_REQUESTS:
        slow_filepath = "reports/slow_requests_history.csv"
        try:
            with open(slow_filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "RequestID", "Endpoint", "LatencyMS"])
                for timestamp, req_id, endpoint, latency in SLOW_REQUESTS:
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                    writer.writerow([time_str, req_id, endpoint, latency])
            logger.info(f"Successfully exported {len(SLOW_REQUESTS)} slow request logs to: {os.path.abspath(slow_filepath)}")
        except Exception as e:
            logger.error(f"Failed to export CSV slow request logs: {str(e)}")
