from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.common import BaseResponse
from app.schemas.detect import DetectionItem
from app.schemas.perception import PerceptionGraph

class HazardItem(BaseModel):
    label: str = Field(..., description="The name of the hazardous object class")
    box: List[float] = Field(..., description="Bounding box coordinate representing [xmin, ymin, xmax, ymax]")
    type: str = Field(..., description="Type of hazard, e.g. tripping_risk, collision_risk, burn_risk")
    severity: str = Field(..., description="Severity level, e.g. low, medium, high")
    description: str = Field(..., description="A friendly description of why this object is a hazard")

class SceneAnalysisResult(BaseResponse):
    caption: str = Field(..., description="Deep visual caption of the scene")
    objects: List[DetectionItem] = Field(default_factory=list, description="All detected objects in the scene")
    hazards: List[HazardItem] = Field(default_factory=list, description="Identified hazards in the scene")
    narration: str = Field(..., description="Accessibility-focused narration text")
    perception_graph: Optional[PerceptionGraph] = Field(default=None, description="Structured perception graph layer")
