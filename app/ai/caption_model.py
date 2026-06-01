import time
from PIL import Image
from app.ai.base import BaseInferenceWrapper
from app.core.exceptions import ModelInferenceError
import logging

logger = logging.getLogger("accessvision")

class CaptionModelWrapper(BaseInferenceWrapper):
    """Wrapper for Image Captioning models (e.g., Salesforce/blip-image-captioning-base)."""
    
    def _load_actual_model(self) -> None:
        try:
            # Defensive imports - only required if INFERENCE_MODE is local
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device} for image captioning")
            
            self.processor = BlipProcessor.from_pretrained(self.model_path)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
        except ImportError as e:
            logger.error(
                "Missing ML dependencies (torch/transformers) for actual captioning inference. "
                "Ensure they are installed or run in 'mock' mode."
            )
            raise ModelInferenceError(f"Failed to load model due to missing packages: {str(e)}")
        except Exception as e:
            raise ModelInferenceError(f"Error loading captioning model: {str(e)}")

    def _unload_actual_model(self) -> None:
        import torch
        self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def predict(self, image: Image.Image) -> str:
        """Generates a caption text for the provided PIL Image.
        
        This call is synchronous and blocking. Run inside a thread pool in the service layer.
        """
        self.load()  # Ensure model is loaded (lazy loading/idempotent)
        
        try:
            import torch
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=50)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Inference failed in CaptionModelWrapper: {str(e)}")
            raise ModelInferenceError(f"Captioning inference failed: {str(e)}")
