from fastapi import APIRouter, File, UploadFile, Form, Depends
from app.services.vqa_service import VqaService
from app.schemas.vqa import VqaResult, VqaJsonRequest
from app.api.dependencies import verify_api_key

router = APIRouter()

@router.post(
    "/ask",
    response_model=VqaResult,
    summary="Ask a question about an image (Form)",
    description="Upload an image and submit a textual question about it using multipart form-data."
)
async def ask_question_form(
    file: UploadFile = File(..., description="Target image file (e.g. JPEG, PNG)"),
    question: str = Form(..., description="Question to answer about the image"),
    service: VqaService = Depends(VqaService),
    api_key: str = Depends(verify_api_key)
):
    # Read and preprocess raw file bytes once at entry point to optimize size/format
    from app.utils.image import preprocess_image_bytes
    image_bytes = await file.read()
    optimized_bytes = preprocess_image_bytes(image_bytes)
    
    # Process VQA request
    result = await service.answer_question(optimized_bytes, question)
    return result

@router.post(
    "/ask-json",
    response_model=VqaResult,
    summary="Ask a question about an image (JSON)",
    description="Submit a question and a base64-encoded image string inside a JSON payload."
)
async def ask_question_json(
    request: VqaJsonRequest,
    service: VqaService = Depends(VqaService),
    api_key: str = Depends(verify_api_key)
):
    # Process Base64 VQA request
    result = await service.answer_question_base64(request.image_base64, request.question)
    return result
