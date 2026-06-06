import asyncio
from typing import Dict, Any, Optional, Union
from PIL import Image
from app.services.base import BaseService
from app.ai.registry import ModelRegistry
from app.core.config import settings
from app.utils.compat import run_in_thread
from app.utils.image import load_image_from_bytes, resize_image_for_model
import logging

logger = logging.getLogger("accessvision")

class FlorenceService(BaseService):
    """Business logic service for handling Florence-2 Grounded Multimodal tasks."""
    
    def __init__(self):
        self.model_wrapper = ModelRegistry.get_florence_wrapper()

    async def run_task(
        self, 
        image_input: Union[bytes, Image.Image], 
        task: str, 
        text_input: Optional[str] = None
    ) -> Dict[str, Any]:
        """Asynchronously runs a Florence-2 task on raw image bytes or PIL image.
        
        Protected by the Florence concurrency semaphore and executed in a dedicated Florence background thread pool.
        """
        logger.info(f"Starting Florence-2 task: {task} (text_input={text_input})")
        from app.core.telemetry import trace_stage, get_current_telemetry
        
        telemetry = get_current_telemetry()
        
        sem = ModelRegistry.get_florence_semaphore()
        logger.info(f"[QUEUE] Florence jobs active: {sem.active_count}/{settings.FLORENCE_SEMAPHORE_LIMIT}")
        async with sem:
            with trace_stage("FLORENCE"):
                should_close_image = False
                with trace_stage("IMAGE_DECODE"):
                    if isinstance(image_input, bytes):
                        image = load_image_from_bytes(image_input)
                        should_close_image = True
                    else:
                        image = image_input
                    
                with trace_stage("RESIZE"):
                    optimized_image = resize_image_for_model(image, max_size=640)
                    
                with trace_stage("INFERENCE"):
                    # Call predict inside dedicated thread pool
                    result = await run_in_thread(
                        self.model_wrapper.predict, 
                        optimized_image, 
                        task, 
                        text_input,
                        executor=ModelRegistry.get_florence_executor()
                    )
                    
                if optimized_image is not image:
                    optimized_image.close()
                if should_close_image:
                    image.close()
                
        latency_ms = 0.0
        if telemetry and "FLORENCE" in telemetry.timings:
            latency_ms = telemetry.timings["FLORENCE"]
            logger.info(f"[FLORENCE] Task: {task} completed in {latency_ms:.1f}ms")
            
        return result

    async def get_detailed_caption(self, image_input: Union[bytes, Image.Image]) -> str:
        """Convenience method for grounded scene captioning."""
        res = await self.run_task(image_input, "<DETAILED_CAPTION>")
        return res.get("<DETAILED_CAPTION>", "")

    async def get_objects(self, image_input: Union[bytes, Image.Image]) -> Dict[str, Any]:
        """Convenience method for Florence-2 object detection grounding."""
        res = await self.run_task(image_input, "<OD>")
        return res.get("<OD>", {"bboxes": [], "labels": []})

    async def get_phrase_grounding(self, image_input: Union[bytes, Image.Image], phrase: str) -> Dict[str, Any]:
        """Convenience method for phrase grounding within a caption."""
        res = await self.run_task(image_input, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=phrase)
        return res.get("<CAPTION_TO_PHRASE_GROUNDING>", {"bboxes": [], "labels": []})

    async def get_ocr(self, image_input: Union[bytes, Image.Image]) -> str:
        """Convenience method for optical character recognition (OCR)."""
        res = await self.run_task(image_input, "<OCR>")
        return res.get("<OCR>", "")
