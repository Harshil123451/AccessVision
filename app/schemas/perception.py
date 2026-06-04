from typing import List
from pydantic import BaseModel, Field, ConfigDict

class GroundedObject(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True
    )

    class_name: str = Field(..., serialization_alias="class", validation_alias="class", description="The object category name")
    confidence: float = Field(..., description="The detection or grounding confidence score (0.0 to 1.0)")
    color: str = Field(..., description="Identified color of the object")
    position: str = Field(..., description="Grounded spatial position descriptor")
    size: str = Field(..., description="Categorized size based on bounding box area")
    grounding_source: str = Field(..., serialization_alias="grounding source", validation_alias="grounding source", description="The sensor/model that detected the object (e.g. YOLO, Florence, YOLO+Florence)")
    narration_confidence: str = Field(..., serialization_alias="narration confidence", validation_alias="narration confidence", description="Accessibility narration confidence level (HIGH, MEDIUM, LOW)")

class PerceptionGraph(BaseModel):
    objects: List[GroundedObject] = Field(default_factory=list, description="Graph of visually grounded and tracked objects in the current scene")
