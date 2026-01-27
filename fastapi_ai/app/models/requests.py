from pydantic import BaseModel, HttpUrl, Field
from typing import Optional

class OCRRequest(BaseModel):
    """OCR Request Schema"""
    image_id: str = Field(..., description="Unique image identifier")
    user_id: str = Field(..., description="User identifier")
    image_url: HttpUrl = Field(..., description="Image URL to process")
    callback_url: Optional[HttpUrl] = Field(None, description="Optional callback URL")
    
    # Optional overrides
    use_layout_detection: Optional[bool] = None
    use_chart_recognition: Optional[bool] = None
    use_llm_enhancement: Optional[bool] = None