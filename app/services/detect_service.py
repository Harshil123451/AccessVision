import asyncio
from typing import Optional, Union
from PIL import Image
from app.services.base import BaseService
from app.ai.registry import ModelRegistry
from app.core.config import settings
from app.utils.compat import run_in_thread
from app.utils.image import load_image_from_bytes, resize_image_for_model
from app.schemas.detect import DetectionResult, DetectionItem
import logging

logger = logging.getLogger("accessvision")

class DetectService(BaseService):
    """Business logic service for handling Object Detection (YOLO) requests."""

    # Class-level cache mapping md5(image_bytes)_confidence -> DetectionResult
    _cache = {}
    _cache_max_size = 100
    _cache_hits = 0
    _cache_misses = 0

    def __init__(self):
        # Service fetches the singleton wrapper from ModelRegistry
        self.model_wrapper = ModelRegistry.get_yolo_wrapper()

    def _get_cache_key(self, image_bytes: bytes, confidence: float) -> str:
        import hashlib
        h = hashlib.md5(image_bytes).hexdigest()
        return f"{h}_{confidence}"

    @classmethod
    def get_cache_stats(cls) -> dict:
        """Returns statistics on detection cache efficiency."""
        total = cls._cache_hits + cls._cache_misses
        ratio = (cls._cache_hits / total * 100) if total > 0 else 0.0
        return {
            "hits": cls._cache_hits,
            "misses": cls._cache_misses,
            "hit_ratio_percent": round(ratio, 2),
            "size": len(cls._cache)
        }

    async def detect_objects(self, image_input: Union[bytes, Image.Image], confidence: Optional[float] = None) -> DetectionResult:
        """Asynchronously processes raw image bytes or PIL image and runs YOLO object detection.
        
        Uses run_in_thread with a dedicated YOLO executor pool.
        Uses cached detection results if the exact payload was already processed (requires image_input to be bytes).
        """
        if confidence is None:
            confidence = settings.YOLO_CONFIDENCE_THRESHOLD

        logger.info(f"Starting object detection process (conf threshold: {confidence})")
        from app.core.telemetry import trace_stage, get_current_telemetry
        import time
        
        # 1. Check cache first to bypass inference pipeline entirely (only possible if input is bytes)
        has_cached = False
        cache_key = None
        if isinstance(image_input, bytes):
            with trace_stage("CACHE_LOOKUP"):
                cache_key = self._get_cache_key(image_input, confidence)
                has_cached = cache_key in self._cache
            
        if has_cached:
            DetectService._cache_hits += 1
            stats = self.get_cache_stats()
            logger.info(f"[CACHE HIT] reused YOLO detections | hit_ratio={stats['hit_ratio_percent']}%")
            telemetry = get_current_telemetry()
            if telemetry:
                telemetry.cache_hit = True
                telemetry.add_trace("[CACHE HIT] reused YOLO detections")
                telemetry.add_trace(f"[CACHE STATS] hit_ratio={stats['hit_ratio_percent']}%")
            
            cached_result = self._cache[cache_key]
            logger.info(f"[YOLO] Using {settings.YOLO_MODEL_PATH}")
            logger.info(f"[YOLO] Threshold: {confidence}")
            logger.info(f"[YOLO] Raw detections: {len(cached_result.detections)}")
            logger.info(f"[YOLO] Accepted detections (>{confidence}): {len(cached_result.detections)}")
            logger.info("[YOLO] Inference completed in 0.0ms (Cache Hit)")
            return cached_result
            
        if isinstance(image_input, bytes):
            DetectService._cache_misses += 1
            stats = self.get_cache_stats()
            logger.info(f"[CACHE MISS] executing inference | hit_ratio={stats['hit_ratio_percent']}%")
            telemetry = get_current_telemetry()
            if telemetry:
                telemetry.add_trace("[CACHE MISS] executing inference")
                telemetry.add_trace(f"[CACHE STATS] hit_ratio={stats['hit_ratio_percent']}%")

        # 2. Acquire concurrency semaphore to protect CPU inference queue
        async with ModelRegistry.get_yolo_semaphore():
            with trace_stage("YOLO"):
                # 3. Parse and validate image
                should_close_image = False
                with trace_stage("IMAGE_DECODE"):
                    if isinstance(image_input, bytes):
                        image = load_image_from_bytes(image_input)
                        should_close_image = True
                    else:
                        image = image_input
                
                # 4. Optimize image resolution for inference
                with trace_stage("RESIZE"):
                    optimized_image = resize_image_for_model(image, max_size=640)
                
                # 5. Perform inference inside the dedicated YOLO thread pool
                yolo_start = time.perf_counter()
                raw_detections = await run_in_thread(
                    self.model_wrapper.predict, 
                    optimized_image, 
                    confidence_threshold=confidence,
                    executor=ModelRegistry.get_yolo_executor()
                )
                yolo_latency_ms = round((time.perf_counter() - yolo_start) * 1000, 2)
                
                # 6. Clean up PIL images explicitly to save RAM
                if optimized_image is not image:
                    optimized_image.close()
                if should_close_image:
                    image.close()

            # 7. Map raw output list of dicts to schema items, filtering by target confidence
            raw_count = len(raw_detections)
            accepted_detections_raw = [d for d in raw_detections if d["confidence"] >= confidence]
            accepted_count = len(accepted_detections_raw)
            
            detections = [
                DetectionItem(
                    box=d["box"],
                    label=d["label"],
                    confidence=d["confidence"]
                ) for d in accepted_detections_raw
            ]

        # Get latency metrics
        latency_ms = yolo_latency_ms
        logger.info(f"[YOLO] Using {settings.YOLO_MODEL_PATH}")
        logger.info(f"[YOLO] Threshold: {confidence}")
        logger.info(f"[YOLO] Raw detections: {raw_count}")
        logger.info(f"[YOLO] Accepted detections (>{confidence}): {accepted_count}")
        logger.info(f"[YOLO] Inference completed in {latency_ms:.1f}ms")

        result = DetectionResult(
            success=True,
            detections=detections,
            metrics={"inference_ms": latency_ms}
        )

        # 8. Cache result
        if len(self._cache) >= self._cache_max_size:
            logger.info("[CACHE] Evicting all items from YOLO cache (max size reached)")
            self._cache.clear()
        self._cache[cache_key] = result

        return result
