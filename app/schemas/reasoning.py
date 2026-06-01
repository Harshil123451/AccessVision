from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.common import BaseResponse
from app.schemas.detect import DetectionItem

class ReasoningResult(BaseResponse):
    question: str = Field(..., description="The analyzed question")
    intent: str = Field(..., description="The detected intent of the question")
    target_object: Optional[str] = Field(None, description="The detected target object of the question")
    answer: str = Field(..., description="The grounded answer to the question")
    grounded_by: str = Field(..., description="The service or pipeline that grounded the answer")
    detections: List[DetectionItem] = Field(default_factory=list, description="Relevant object detections used for routing/grounding")
