from fastapi import APIRouter
from app.api.v1 import health, caption, detect, vqa, reason, scene

api_router = APIRouter()

# Register sub-routers
api_router.include_router(health.router, tags=["System"])
api_router.include_router(caption.router, prefix="/caption", tags=["Captioning"])
api_router.include_router(detect.router, prefix="/detect", tags=["Object Detection"])
api_router.include_router(vqa.router, prefix="/vqa", tags=["Visual Question Answering"])
api_router.include_router(reason.router, prefix="/reason", tags=["Grounded Reasoning"])
api_router.include_router(scene.router, prefix="/scene", tags=["Scene Understanding"])

