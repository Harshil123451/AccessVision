from typing import List
from pydantic import BaseModel, Field
from app.schemas.common import BaseResponse

class DetectionItem(BaseModel):
    box: List[float] = Field(
        ..., 
        description="Bounding box coordinate representing [xmin, ymin, xmax, ymax]"
    )
    label: str = Field(..., description="The name of the detected object class")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0")

class DetectionResult(BaseResponse):
    detections: List[DetectionItem] = Field(
        default_factory=list, 
        description="List of detected objects found in the image"
    )
