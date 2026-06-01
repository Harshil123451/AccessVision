from PIL import Image
from typing import List
from app.services.base import BaseService
import logging

logger = logging.getLogger("accessvision")

class CropService(BaseService):
    """Service to crop specific regions from an image using YOLO bounding boxes."""

    def crop_object(self, image: Image.Image, box: List[float]) -> Image.Image:
        """Crops a single object region from the PIL image using [xmin, ymin, xmax, ymax]."""
        from app.core.telemetry import trace_stage
        with trace_stage("CROP"):
            width, height = image.size
            # Clamp coordinates to image boundaries
            xmin = max(0, int(box[0]))
            ymin = max(0, int(box[1]))
            xmax = min(width, int(box[2]))
            ymax = min(height, int(box[3]))

            # Check for invalid coordinates
            if xmax <= xmin or ymax <= ymin:
                logger.warning(f"Invalid crop box {box} for image size {image.size}. Returning full image.")
                return image

            return image.crop((xmin, ymin, xmax, ymax))

    def crop_objects(self, image: Image.Image, boxes: List[List[float]]) -> List[Image.Image]:
        """Crops multiple object regions from the PIL image."""
        crops = []
        for box in boxes:
            crops.append(self.crop_object(image, box))
        return crops
