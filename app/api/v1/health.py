from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/health", summary="Perform system health check")
async def health_check():
    """Returns application environment configuration and status."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "inference_mode": settings.INFERENCE_MODE,
        "configuration": {
            "yolo_model": settings.YOLO_MODEL_PATH,
            "caption_model": settings.CAPTION_MODEL_PATH,
            "vqa_model": settings.VQA_MODEL_PATH
        }
    }
