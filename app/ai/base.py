import abc
from typing import Any
import logging
from app.core.config import settings

logger = logging.getLogger("accessvision")

class BaseInferenceWrapper(abc.ABC):
    """Abstract base class for AI Model Inference Wrappers.
    
    Provides standard lifecycle hooks for model loading, prediction,
    and memory cleanup. Can support both mock and actual execution modes.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        logger.info(f"Initialized {self.__class__.__name__} with path '{model_path}'")

    def load(self) -> None:
        """Standard loading lifecycle hook. Ensures the model weights are loaded only once."""
        if self.is_loaded:
            return

        try:
            logger.info(f"Loading actual model weights for {self.__class__.__name__} from {self.model_path}")
            self._load_actual_model()
            self.is_loaded = True
            logger.info(f"Successfully loaded actual model weights for {self.__class__.__name__}")
        except Exception as e:
            logger.exception(f"Failed to load weights for {self.__class__.__name__}: {str(e)}")
            raise e

    def unload(self) -> None:
        """Standard unloading lifecycle hook. Releases memory buffers."""
        if not self.is_loaded:
            return

        try:
            logger.info(f"Unloading weights for {self.__class__.__name__}")
            self._unload_actual_model()
            self.model = None
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to unload model for {self.__class__.__name__}: {str(e)}")


    @abc.abstractmethod
    def _load_actual_model(self) -> None:
        """Subclasses implement this to load actual ML weights (e.g., torch.load, ultralytics.YOLO, transformers)."""
        pass

    @abc.abstractmethod
    def _unload_actual_model(self) -> None:
        """Subclasses implement this to free GPU/VRAM resources (e.g., torch.cuda.empty_cache)."""
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Synchronous prediction interface. Should be run in a separate thread if CPU/GPU bound."""
        pass
