from fastapi import APIRouter, File, UploadFile, Query, Depends
from app.services.detect_service import DetectService
from app.schemas.detect import DetectionResult
from app.api.dependencies import verify_api_key

router = APIRouter()

@router.post(
    "/objects",
    response_model=DetectionResult,
    summary="Detect objects using YOLO",
    description="Upload an image to locate objects, coordinate boxes, class labels, and confidence levels."
)
async def detect_objects(
    file: UploadFile = File(..., description="Target image file (e.g. JPEG, PNG)"),
    confidence: float = Query(default=0.25, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    service: DetectService = Depends(DetectService),
    api_key: str = Depends(verify_api_key)
):
    # Read and preprocess raw file bytes once at entry point to optimize size/format
    from app.utils.image import preprocess_image_bytes
    image_bytes = await file.read()
    optimized_bytes = preprocess_image_bytes(image_bytes)
    
    # Process using detect service
    result = await service.detect_objects(optimized_bytes, confidence=confidence)
    return result
