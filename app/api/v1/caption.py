from fastapi import APIRouter, File, UploadFile, Depends
from app.services.caption_service import CaptionService
from app.schemas.caption import CaptionResult
from app.api.dependencies import verify_api_key

router = APIRouter()

@router.post(
    "/generate", 
    response_model=CaptionResult, 
    summary="Generate image caption",
    description="Upload an image to generate a descriptive alternative text representation."
)
async def generate_caption(
    file: UploadFile = File(..., description="Target image file (e.g. JPEG, PNG)"),
    service: CaptionService = Depends(CaptionService),
    api_key: str = Depends(verify_api_key)
):
    # Read and preprocess raw file bytes once at entry point to optimize size/format
    from app.utils.image import preprocess_image_bytes
    image_bytes = await file.read()
    optimized_bytes = preprocess_image_bytes(image_bytes)
    
    # Process using caption service
    result = await service.generate_caption(optimized_bytes)
    return result
