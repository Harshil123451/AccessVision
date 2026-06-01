from pydantic import BaseModel, Field

class TimeMetrics(BaseModel):
    inference_ms: float = Field(..., description="Inference latency in milliseconds")

class BaseResponse(BaseModel):
    success: bool = Field(True, description="Indicates if the request was processed successfully")
    metrics: TimeMetrics = Field(..., description="Execution performance metrics")
