import time
from typing import List, Dict, Any
from PIL import Image
from app.ai.base import BaseInferenceWrapper
from app.core.exceptions import ModelInferenceError
import logging

logger = logging.getLogger("accessvision")

class YoloModelWrapper(BaseInferenceWrapper):
    """Wrapper for YOLO object detection models (e.g., Ultralytics YOLOv8)."""

    def _load_actual_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError as e:
            logger.error(
                "Missing 'ultralytics' dependency for YOLO detection. "
                "Ensure it is installed or run in 'mock' mode."
            )
            raise ModelInferenceError(f"Failed to load model due to missing packages: {str(e)}")
        except Exception as e:
            raise ModelInferenceError(f"Error loading YOLO model: {str(e)}")

    def _unload_actual_model(self) -> None:
        # Ultralytics doesn't expose explicit unload easily, garbage collection cleans it.
        pass

    def predict(self, image: Image.Image, confidence_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Detects objects in the provided PIL Image.
        
        Returns a list of dicts:
            [
                {
                    "box": [xmin, ymin, xmax, ymax],  # normalized or pixel coordinates
                    "label": "class_name",
                    "confidence": float
                }
            ]
        """
        self.load()
        
        try:
            # Perform inference using YOLO
            results = self.model(image, conf=confidence_threshold, verbose=False)
            detections = []
            
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for box in boxes:
                    # Get coordinates
                    # xyxy is pixel coords [xmin, ymin, xmax, ymax]
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    label = result.names[cls_id]
                    
                    detections.append({
                        "box": [round(coord, 2) for coord in xyxy],
                        "label": label,
                        "confidence": round(conf, 4)
                    })
                    
            return detections
        except Exception as e:
            logger.error(f"Inference failed in YoloModelWrapper: {str(e)}")
            raise ModelInferenceError(f"YOLO object detection failed: {str(e)}")
