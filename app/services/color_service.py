import numpy as np
import colorsys
from PIL import Image
from app.services.base import BaseService
from typing import Tuple
import logging

logger = logging.getLogger("accessvision")

class ColorService(BaseService):
    """Service to extract dominant color from cropped image regions and map to human-readable names."""

    # Pre-defined RGB standard colors for distance mapping
    COLOR_MAP = {
        "red": (220, 20, 60),
        "orange": (255, 140, 0),
        "yellow": (255, 215, 0),
        "green": (34, 139, 34),
        "blue": (0, 0, 205),
        "purple": (128, 0, 128),
        "pink": (255, 105, 180),
        "brown": (139, 69, 19),
        "white": (255, 255, 255),
        "black": (20, 20, 20),
        "grey": (128, 128, 128)
    }

    def _rgb_to_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Converts RGB tuple to human-readable color name using hybrid HSV + Euclidean RGB distance."""
        r, g, b = rgb
        # Convert RGB to normalized values for HSV conversion
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

        # 1. Broad HSV classifications (especially for white, black, gray)
        if v < 0.18:
            return "black"
        if s < 0.15:
            if v > 0.8:
                return "white"
            return "grey"

        # 2. Euclidean distance in RGB space to standard anchor colors for high precision
        min_dist = float("inf")
        closest_color = "unknown"
        for name, target_rgb in self.COLOR_MAP.items():
            dist = np.sqrt((r - target_rgb[0])**2 + (g - target_rgb[1])**2 + (b - target_rgb[2])**2)
            if dist < min_dist:
                min_dist = dist
                closest_color = name

        return closest_color

    def get_dominant_color(self, crop_image: Image.Image) -> str:
        """Extracts the dominant color of the cropped image.
        
        To avoid background colors at borders, it focuses on the center 50% of the crop.
        """
        from app.core.telemetry import trace_stage
        with trace_stage("COLOR"):
            # Convert to RGB if not already
            if crop_image.mode != "RGB":
                crop_image = crop_image.convert("RGB")

            # Crop the center 50% region
            w, h = crop_image.size
            left = int(w * 0.25)
            top = int(h * 0.25)
            right = int(w * 0.75)
            bottom = int(h * 0.75)
            
            # Ensure it is at least 1x1
            if right <= left:
                right = left + 1
            if bottom <= top:
                bottom = top + 1

            center_crop = crop_image.crop((left, top, right, bottom))
            
            # Resize center_crop to 1x1 to average the pixels
            one_pixel = center_crop.resize((1, 1), Image.Resampling.BILINEAR)
            pixel = one_pixel.getpixel((0, 0))
            rgb = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
            
            color_name = self._rgb_to_color_name(rgb)
            logger.info(f"Extracted average RGB {rgb} from center region, mapped to: {color_name}")
            return color_name
