import asyncio
import base64
from app.services.base import BaseService
from app.ai.registry import ModelRegistry
from app.core.config import settings
from app.core.exceptions import InvalidImageError
from app.utils.compat import run_in_thread
from app.utils.image import load_image_from_bytes, resize_image_for_model
from app.schemas.vqa import VqaResult
import logging

logger = logging.getLogger("accessvision")

class VqaService(BaseService):
    """Business logic service for handling Visual Question Answering (VQA) requests."""

    def __init__(self):
        # Service fetches the singleton wrapper from ModelRegistry
        self.model_wrapper = ModelRegistry.get_vqa_wrapper()

    async def answer_question(self, image_bytes: bytes, question: str) -> VqaResult:
        """Asynchronously processes raw image bytes and answers a specific visual question."""
        logger.info(f"Starting VQA query process. Question: '{question}'")
        from app.core.telemetry import trace_stage, get_current_telemetry

        telemetry = get_current_telemetry()
        async with ModelRegistry.get_semaphore():
            with trace_stage("VQA"):
                # 1. Parse and validate image
                with trace_stage("IMAGE_DECODE"):
                    image = load_image_from_bytes(image_bytes)
                
                # 2. Optimize image resolution for inference
                with trace_stage("RESIZE"):
                    optimized_image = resize_image_for_model(image, max_size=640)
                
                # 3. Perform inference inside thread pool
                answer = await run_in_thread(
                    self.model_wrapper.predict, 
                    optimized_image, 
                    question
                )
                
                # 4. Clean up PIL images explicitly to save RAM
                if optimized_image is not image:
                    optimized_image.close()
                image.close()

        # Get latency metrics if available
        latency_ms = 0.0
        if telemetry and "VQA" in telemetry.timings:
            latency_ms = telemetry.timings["VQA"]
            logger.info(f"[VQA] Inference completed in {latency_ms}ms")

        return VqaResult(
            success=True,
            question=question,
            answer=answer,
            metrics={"inference_ms": latency_ms}
        )

    async def answer_question_base64(self, image_base64: str, question: str) -> VqaResult:
        """Helper to process base64-encoded image strings directly (e.g. for JSON API clients)."""
        try:
            # Clean base64 header if present (e.g. 'data:image/jpeg;base64,...')
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            logger.error(f"Failed to decode base64 image input: {str(e)}")
            raise InvalidImageError("Provided image_base64 string is not valid base64.")

        # Preprocess base64 decoded bytes once at entry point to optimize format and size
        from app.utils.image import preprocess_image_bytes
        optimized_bytes = preprocess_image_bytes(image_bytes)

        return await self.answer_question(optimized_bytes, question)
