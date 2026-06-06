import time
from typing import Any, Dict, Optional
from PIL import Image
from app.ai.base import BaseInferenceWrapper
from app.core.exceptions import ModelInferenceError
import logging

logger = logging.getLogger("accessvision")

class FlorenceModelWrapper(BaseInferenceWrapper):
    """Wrapper for Florence-2 multimodal grounding models (e.g., microsoft/Florence-2-base)."""
    
    def _load_actual_model(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device} for Florence-2")
            
            # Florence-2 models require trust_remote_code=True
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            if hasattr(config, "text_config"):
                if not hasattr(config.text_config, "forced_bos_token_id"):
                    config.text_config.forced_bos_token_id = getattr(config.text_config, "bos_token_id", 0)
                    logger.info("[PATCH] Injected missing forced_bos_token_id into Florence2LanguageConfig")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                config=config
            ).to(self.device)
            self._warmup()
        except ImportError as e:
            logger.error(
                "Missing ML dependencies (torch/transformers/einops/timm) for Florence-2. "
                "Ensure they are installed."
            )
            raise ModelInferenceError(f"Failed to load Florence-2 model due to missing packages: {str(e)}")
        except Exception as e:
            raise ModelInferenceError(f"Error loading Florence-2 model: {str(e)}")

    def _warmup(self) -> None:
        """Runs a dummy inference pass at startup to warm up Florence-2 execution kernels."""
        try:
            logger.info("Running Florence-2 model warmup pass...")
            import torch
            # Create a black dummy image
            dummy_image = Image.new("RGB", (640, 640), color=0)
            task = "<CAPTION>"
            inputs = self.processor(text=task, images=dummy_image, return_tensors="pt").to(self.device)
            inference_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
            with inference_ctx:
                _ = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=10,
                    num_beams=1
                )
            logger.info("Florence-2 model warmup pass completed successfully.")
        except Exception as e:
            logger.warning(f"Florence-2 model warmup pass failed: {str(e)}")

    def _unload_actual_model(self) -> None:
        import torch
        self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def predict(self, image: Image.Image, task: str = "<DETAILED_CAPTION>", text_input: Optional[str] = None) -> Any:
        """Runs the Florence-2 model on the provided image and task prompt.
        
        This call is synchronous and blocking. Run inside a thread pool in the service layer.
        """
        self.load() # Idempotent load
        
        try:
            import torch
            
            # Construct the prompt
            prompt = f"{task}{text_input}" if text_input else task
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
                
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Post-process generation
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task, 
                image_size=(image.width, image.height)
            )
            return parsed_answer
        except Exception as e:
            logger.error(f"Inference failed in FlorenceModelWrapper: {str(e)}")
            raise ModelInferenceError(f"Florence-2 inference failed: {str(e)}")
