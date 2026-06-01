import asyncio
from app.services.base import BaseService
from app.ai.registry import ModelRegistry
from app.core.config import settings
from app.utils.compat import run_in_thread
from app.utils.image import load_image_from_bytes, resize_image_for_model
from app.schemas.caption import CaptionResult
import logging

logger = logging.getLogger("accessvision")

class CaptionService(BaseService):
    """Business logic service for handling Image Captioning requests."""
    
    def __init__(self):
        # Service fetches the singleton wrapper from ModelRegistry
        self.model_wrapper = ModelRegistry.get_caption_wrapper()
        
    async def generate_caption(self, image_bytes: bytes) -> CaptionResult:
        """Asynchronously processes the raw image bytes and generates a textual description.
        
        Uses run_in_thread to run synchronous ML inference in a non-blocking worker thread.
        Protected by an async concurrency semaphore.
        """
        logger.info("Starting caption generation process")
        from app.core.telemetry import trace_stage, get_current_telemetry

        telemetry = get_current_telemetry()
        async with ModelRegistry.get_semaphore():
            with trace_stage("BLIP"):
                # 1. Parse and validate image
                with trace_stage("IMAGE_DECODE"):
                    image = load_image_from_bytes(image_bytes)
                
                # 2. Optimize image resolution for inference
                with trace_stage("RESIZE"):
                    optimized_image = resize_image_for_model(image, max_size=640)
                
                # 3. Perform inference inside thread pool to prevent blocking the event loop
                caption = await run_in_thread(self.model_wrapper.predict, optimized_image)
                
                # 4. Clean up PIL images explicitly to save RAM
                if optimized_image is not image:
                    optimized_image.close()
                image.close()
                
        # Get latency metrics if available
        latency_ms = 0.0
        if telemetry and "BLIP" in telemetry.timings:
            latency_ms = telemetry.timings["BLIP"]
            logger.info(f"[BLIP] Inference completed in {latency_ms}ms")
            
        return CaptionResult(
            success=True,
            caption=caption,
            metrics={"inference_ms": latency_ms}
        )
