import io
from PIL import Image
from app.core.exceptions import InvalidImageError
import logging

logger = logging.getLogger("accessvision")

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Loads and validates a PIL Image from raw binary bytes.
    
    Raises:
        InvalidImageError: If the image cannot be decoded.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Verify it is a valid image file by reading properties
        image.verify()
        # Re-open after verify() since verify() closes the file pointer/resets stream
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB (keeps representation consistent and prevents alpha channel issues in some models)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Failed to decode image: {str(e)}")
        raise InvalidImageError("Uploaded file is not a valid image or is corrupt.")

def resize_image_for_model(image: Image.Image, max_size: int = 640) -> Image.Image:
    """Resizes image keeping aspect ratio, limiting maximum dimension.
    Used to optimize network transfer and model speed.
    """
    w, h = image.size
    if max_size >= max(w, h):
        return image
        
    if w > h:
        new_w = max_size
        new_h = int(h * (max_size / w))
    else:
        new_h = max_size
        new_w = int(w * (max_size / h))
        
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def save_image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """Converts a PIL Image back into raw bytes (e.g., for sending to model inference)."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()

def preprocess_image_bytes(image_bytes: bytes) -> bytes:
    """Preprocesses raw uploaded image bytes by decoding, resizing (to MAX_IMAGE_SIZE),
    converting to optimized RGB JPEG (at JPEG_QUALITY), and returning optimized bytes.
    
    This runs ONCE per request at the API entry point to prevent duplicated decoding/resizing.
    """
    import time
    from app.core.config import settings
    from app.core.telemetry import trace_stage
    
    start_time = time.perf_counter()
    try:
        with trace_stage("PREPROCESS"):
            with trace_stage("IMAGE_DECODE"):
                img = Image.open(io.BytesIO(image_bytes))
                # Convert to RGB to ensure JPEG compatibility and prevent alpha channel issues
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
            
            with trace_stage("RESIZE"):
                # Resize
                w, h = img.size
                max_size = settings.MAX_IMAGE_SIZE
                if max_size < max(w, h):
                    if w > h:
                        new_w = max_size
                        new_h = int(h * (max_size / w))
                    else:
                        new_h = max_size
                        new_w = int(w * (max_size / h))
                    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                else:
                    new_w, new_h = w, h
                    img_resized = img
            
            with trace_stage("JPEG_OPTIMIZE"):
                # Save as optimized JPEG in memory
                buf = io.BytesIO()
                img_resized.save(buf, format="JPEG", quality=settings.JPEG_QUALITY, optimize=True)
                optimized_bytes = buf.getvalue()
                
            # Cleanup PIL resources
            if img_resized is not img:
                img_resized.close()
            img.close()
            
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(f"[PREPROCESS] Resized image from {w}x{h} to {new_w}x{new_h} and optimized to JPEG (Quality: {settings.JPEG_QUALITY}) in {elapsed_ms}ms. Size reduced from {len(image_bytes)} to {len(optimized_bytes)} bytes.")
        return optimized_bytes
    except Exception as e:
        logger.error(f"Failed to preprocess image: {str(e)}")
        raise InvalidImageError("Uploaded file is not a valid image or is corrupt.")

