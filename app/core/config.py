import os
from typing import List, Literal, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    APP_NAME: str = "AccessVision API"
    APP_ENV: Literal["development", "production", "testing"] = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "127.0.0.1"

    # AI Engine Configuration
    INFERENCE_MODE: Literal["local"] = "local"
    YOLO_MODEL_PATH: str = "yolov8s.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    CAPTION_MODEL_PATH: str = "Salesforce/blip-image-captioning-base"
    VQA_MODEL_PATH: str = "Salesforce/blip-vqa-base"
    FLORENCE_MODEL_PATH: str = "microsoft/Florence-2-base"
    MIGRATION_STAGE: int = 2
    FLORENCE_CONFIDENCE_THRESHOLD: float = 0.5
    FLORENCE_SEMAPHORE_LIMIT: int = 2
    YOLO_SEMAPHORE_LIMIT: int = 10
    FLORENCE_TIMEOUT: float = 3.0
    SCENE_SIMILARITY_THRESHOLD: float = 15.0
    SCENE_STATIC_THRESHOLD: float = 8.0
    SCENE_IOU_THRESHOLD: float = 0.6

    # Performance Optimizations
    MAX_IMAGE_SIZE: int = 640
    JPEG_QUALITY: int = 80
    SEMAPHORE_LIMIT: int = 4

    # Security
    API_KEY_SECRET: str = "development-secret-key"
    ALLOWED_ORIGINS: List[str] = ["*"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            # Check if it looks like a JSON list
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except ValueError:
                    pass
            return [i.strip() for i in v.split(",")]
        return v

settings = Settings()
