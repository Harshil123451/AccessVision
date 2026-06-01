from pydantic import Field
from app.schemas.common import BaseResponse

class CaptionResult(BaseResponse):
    caption: str = Field(..., description="The generated description for the uploaded image")
