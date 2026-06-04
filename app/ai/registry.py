from typing import Dict
from app.ai.base import BaseInferenceWrapper
from app.ai.caption_model import CaptionModelWrapper
from app.ai.yolo_model import YoloModelWrapper
from app.ai.vqa_model import VqaModelWrapper
from app.ai.florence_model import FlorenceModelWrapper
from app.core.config import settings
import logging

logger = logging.getLogger("accessvision")

class ModelRegistry:
    """Manages lifecycles and singleton instances of all AI Inference wrappers.
    
    Prevents duplicate loading of heavy model weights into memory.
    """
    _instances: Dict[str, BaseInferenceWrapper] = {}

    @classmethod
    def get_caption_wrapper(cls) -> CaptionModelWrapper:
        key = f"caption_{settings.CAPTION_MODEL_PATH}"
        if key not in cls._instances:
            cls._instances[key] = CaptionModelWrapper(settings.CAPTION_MODEL_PATH)
        return cls._instances[key]

    @classmethod
    def get_yolo_wrapper(cls) -> YoloModelWrapper:
        key = f"yolo_{settings.YOLO_MODEL_PATH}"
        if key not in cls._instances:
            cls._instances[key] = YoloModelWrapper(settings.YOLO_MODEL_PATH)
        return cls._instances[key]

    @classmethod
    def get_vqa_wrapper(cls) -> VqaModelWrapper:
        key = f"vqa_{settings.VQA_MODEL_PATH}"
        if key not in cls._instances:
            cls._instances[key] = VqaModelWrapper(settings.VQA_MODEL_PATH)
        return cls._instances[key]

    @classmethod
    def get_florence_wrapper(cls) -> FlorenceModelWrapper:
        key = f"florence_{settings.FLORENCE_MODEL_PATH}"
        if key not in cls._instances:
            cls._instances[key] = FlorenceModelWrapper(settings.FLORENCE_MODEL_PATH)
        return cls._instances[key]

    @classmethod
    def load_all(cls) -> None:
        """Pre-loads all models into memory. Call during app startup."""
        logger.info(f"Pre-loading all registered models (Migration Stage: {settings.MIGRATION_STAGE})...")
        cls.get_yolo_wrapper().load()
        cls.get_florence_wrapper().load()
        if settings.MIGRATION_STAGE < 4:
            cls.get_caption_wrapper().load()
            cls.get_vqa_wrapper().load()
        else:
            logger.info("Retirement Stage 4 active: Skipping BLIP pre-loading.")
        logger.info("All registered models pre-loaded successfully.")

    @classmethod
    def unload_all(cls) -> None:
        """Unloads all models from memory. Call during app shutdown."""
        logger.info("Unloading all models...")
        for key, wrapper in cls._instances.items():
            try:
                wrapper.unload()
            except Exception as e:
                logger.error(f"Error unloading model wrapper {key}: {str(e)}")
        cls._instances.clear()
        logger.info("All models unloaded.")

    _semaphore = None
    _semaphore_loop = None
    _florence_semaphore = None
    _florence_semaphore_loop = None
    _yolo_semaphore = None
    _yolo_semaphore_loop = None

    @classmethod
    def get_semaphore(cls):
        """Returns the global ObservableSemaphore singleton, dynamically bound to the current running loop to prevent deadlocks."""
        import asyncio
        from app.core.config import settings
        from app.core.telemetry import ObservableSemaphore
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
            
        if cls._semaphore is None or cls._semaphore_loop is not current_loop:
            cls._semaphore = ObservableSemaphore(settings.SEMAPHORE_LIMIT)
            cls._semaphore_loop = current_loop
        return cls._semaphore

    @classmethod
    def get_florence_semaphore(cls):
        """Returns the Florence-specific concurrency semaphore, dynamically bound to the current loop."""
        import asyncio
        from app.core.config import settings
        from app.core.telemetry import ObservableSemaphore
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
            
        if cls._florence_semaphore is None or cls._florence_semaphore_loop is not current_loop:
            cls._florence_semaphore = ObservableSemaphore(settings.FLORENCE_SEMAPHORE_LIMIT)
            cls._florence_semaphore_loop = current_loop
        return cls._florence_semaphore

    @classmethod
    def get_yolo_semaphore(cls):
        """Returns the YOLO-specific high-concurrency semaphore, dynamically bound to the current loop."""
        import asyncio
        from app.core.config import settings
        from app.core.telemetry import ObservableSemaphore
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
            
        if cls._yolo_semaphore is None or cls._yolo_semaphore_loop is not current_loop:
            cls._yolo_semaphore = ObservableSemaphore(settings.YOLO_SEMAPHORE_LIMIT)
            cls._yolo_semaphore_loop = current_loop
        return cls._yolo_semaphore
