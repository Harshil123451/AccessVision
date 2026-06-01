import time
from PIL import Image
from app.ai.base import BaseInferenceWrapper
from app.core.exceptions import ModelInferenceError
import logging

logger = logging.getLogger("accessvision")

class VqaModelWrapper(BaseInferenceWrapper):
    """Wrapper for Visual Question Answering (VQA) models (e.g., Salesforce/blip-vqa-base)."""

    def _load_actual_model(self) -> None:
        try:
            import torch
            from transformers import BlipProcessor, BlipForQuestionAnswering
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device} for VQA")
            
            self.processor = BlipProcessor.from_pretrained(self.model_path)
            self.model = BlipForQuestionAnswering.from_pretrained(self.model_path).to(self.device)
        except ImportError as e:
            logger.error(
                "Missing ML dependencies (torch/transformers) for VQA. "
                "Ensure they are installed or run in 'mock' mode."
            )
            raise ModelInferenceError(f"Failed to load model due to missing packages: {str(e)}")
        except Exception as e:
            raise ModelInferenceError(f"Error loading VQA model: {str(e)}")

    def _unload_actual_model(self) -> None:
        import torch
        self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def predict(self, image: Image.Image, question: str) -> str:
        """Answers a question about the provided PIL Image.
        
        This call is synchronous and blocking. Run inside a thread pool in the service layer.
        """
        self.load()
        
        try:
            import torch
            inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=50)
            
            answer = self.processor.decode(out[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            logger.error(f"Inference failed in VqaModelWrapper: {str(e)}")
            raise ModelInferenceError(f"VQA inference failed: {str(e)}")
