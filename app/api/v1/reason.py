from fastapi import APIRouter, File, UploadFile, Form, Depends
from typing import Optional
from app.services.question_router_service import QuestionRouterService
from app.schemas.reasoning import ReasoningResult
from app.api.dependencies import verify_api_key

router = APIRouter()

@router.post(
    "/query",
    response_model=ReasoningResult,
    summary="Submit a query to the Grounded Multimodal Reasoning Pipeline",
    description="Submit an image and a question. The pipeline will route the question, detect targets, crop regions, and use the best service to answer without hallucinations."
)
async def query_pipeline(
    file: UploadFile = File(..., description="Target image file"),
    question: str = Form(..., description="Question to answer"),
    is_mirrored: bool = Form(default=False, description="Whether the camera preview was mirrored"),
    session_id: Optional[str] = Form(default=None, description="Unique session ID for cache isolation and tracking"),
    service: QuestionRouterService = Depends(QuestionRouterService),
    api_key: str = Depends(verify_api_key)
):
    # Read and preprocess raw file bytes once at entry point to optimize size/format
    from app.utils.image import preprocess_image_bytes
    image_bytes = await file.read()
    optimized_bytes = preprocess_image_bytes(image_bytes, is_mirrored=is_mirrored)
    
    result = await service.route_and_reason(optimized_bytes, question, is_mirrored=is_mirrored, session_id=session_id)
    return result
