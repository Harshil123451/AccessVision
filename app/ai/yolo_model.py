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
            self._warmup()
        except ImportError as e:
            logger.error(
                "Missing 'ultralytics' dependency for YOLO detection. "
                "Ensure it is installed or run in 'mock' mode."
            )
            raise ModelInferenceError(f"Failed to load model due to missing packages: {str(e)}")
        except Exception as e:
            raise ModelInferenceError(f"Error loading YOLO model: {str(e)}")

    def _warmup(self) -> None:
        """Runs a dummy inference pass at startup to warm up CPU execution kernels."""
        try:
            logger.info("Running YOLO model warmup pass...")
            import torch
            # Create a 640x640 dummy image
            dummy_image = Image.new("RGB", (640, 640), color=0)
            inference_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
            with inference_ctx:
                _ = self.model(dummy_image, conf=0.25, verbose=False)
            logger.info("YOLO model warmup pass completed successfully.")
        except Exception as e:
            logger.warning(f"YOLO model warmup pass failed: {str(e)}")

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
            import torch
            
            # Execute inference under inference mode context to disable gradient tracking
            inference_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
            with inference_ctx:
                results = self.model(image, conf=confidence_threshold, verbose=False)
                
            detections = []
            
            if len(results) > 0:
                result = results[0]
                
                # Retrieve Ultralytics internal sub-stage metrics (in ms)
                preprocess_ms = result.speed.get("preprocess", 0.0)
                forward_ms = result.speed.get("inference", 0.0)
                postprocess_ms = result.speed.get("postprocess", 0.0)
                
                parse_start = time.perf_counter()
                boxes = result.boxes
                
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    label = result.names[cls_id]
                    
                    detections.append({
                        "box": [round(coord, 2) for coord in xyxy],
                        "label": label,
                        "confidence": round(conf, 4)
                    })
                    
                parse_ms = (time.perf_counter() - parse_start) * 1000
                
                # Record to request-scoped telemetry
                from app.core.telemetry import get_current_telemetry
                telemetry = get_current_telemetry()
                if telemetry:
                    telemetry.record_timing("YOLO_PREPROCESS", preprocess_ms)
                    telemetry.record_timing("YOLO_FORWARD", forward_ms)
                    telemetry.record_timing("YOLO_POSTPROCESS", postprocess_ms)
                    telemetry.record_timing("YOLO_PARSE", parse_ms)
                    
                    telemetry.add_trace(f"[YOLO_PREPROCESS] {preprocess_ms:.1f}ms")
                    telemetry.add_trace(f"[YOLO_FORWARD] {forward_ms:.1f}ms")
                    telemetry.add_trace(f"[YOLO_POSTPROCESS] {postprocess_ms:.1f}ms")
                    telemetry.add_trace(f"[YOLO_PARSE] {parse_ms:.1f}ms")
                    
            return detections
        except Exception as e:
            logger.error(f"Inference failed in YoloModelWrapper: {str(e)}")
            raise ModelInferenceError(f"YOLO object detection failed: {str(e)}")
