from PIL import Image
from typing import List
from app.services.base import BaseService
import logging

logger = logging.getLogger("accessvision")

class CropService(BaseService):
    """Service to crop specific regions from an image using YOLO bounding boxes."""

    def crop_object(self, image: Image.Image, box: List[float], margin: float = -0.05) -> Image.Image:
        """Crops a single object region from the PIL image using [xmin, ymin, xmax, ymax].
        
        Supports configurable crop margins (fraction of width/height).
        A negative margin shrinks the box (inset) to avoid background contamination.
        A positive margin pads the box.
        """
        from app.core.telemetry import trace_stage
        with trace_stage("CROP"):
            width, height = image.size
            xmin, ymin, xmax, ymax = box
            
            box_w = xmax - xmin
            box_h = ymax - ymin
            
            # Apply margin adjustments
            if margin != 0.0:
                xmin = xmin - (box_w * margin)
                xmax = xmax + (box_w * margin)
                ymin = ymin - (box_h * margin)
                ymax = ymax + (box_h * margin)
            
            # Clamp coordinates to image boundaries
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(width, int(xmax))
            ymax = min(height, int(ymax))

            # Check for invalid coordinates
            if xmax <= xmin or ymax <= ymin:
                logger.warning(f"Invalid crop box {box} with margin {margin} for image size {image.size}. Returning full image.")
                return image

            return image.crop((xmin, ymin, xmax, ymax))

    def crop_objects(self, image: Image.Image, boxes: List[List[float]], margin: float = -0.05) -> List[Image.Image]:
        """Crops multiple object regions from the PIL image."""
        crops = []
        for box in boxes:
            crops.append(self.crop_object(image, box, margin))
        return crops
