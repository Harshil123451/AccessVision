from fastapi import APIRouter, File, UploadFile, Depends, Form
from typing import Optional
from app.services.scene_service import SceneService
from app.schemas.scene import SceneAnalysisResult
from app.api.dependencies import verify_api_key

router = APIRouter()

@router.post(
    "/analyze",
    response_model=SceneAnalysisResult,
    summary="Generate unified scene understanding",
    description="Analyzes the image using concurrent YOLO detection and BLIP/Florence-2 captioning, assess potential environmental hazards, and produces descriptive accessibility narration."
)
async def analyze_scene(
    file: UploadFile = File(..., description="Target image file"),
    is_mirrored: bool = Form(default=False, description="Whether the camera preview was mirrored"),
    mode: str = Form(default="fast", description="Inference mode: fast, medium, or slow"),
    session_id: Optional[str] = Form(default=None, description="Unique session ID for cache isolation and tracking"),
    service: SceneService = Depends(SceneService),
    api_key: str = Depends(verify_api_key)
):
    # Read and preprocess raw file bytes once at entry point to optimize size/format
    from app.utils.image import preprocess_image_bytes
    image_bytes = await file.read()
    optimized_bytes = preprocess_image_bytes(image_bytes, is_mirrored=is_mirrored)
    
    result = await service.analyze_scene(optimized_bytes, is_mirrored=is_mirrored, mode=mode, session_id=session_id)
    return result

@router.get(
    "/memory",
    summary="Get recent scene memory",
    description="Retrieve the history of the last 10 analyzed scenes."
)
async def get_scene_memory(
    session_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    return SceneService.get_recent_memory(session_id)
