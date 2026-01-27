from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ProcessingStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_VALIDATION = "needs_validation"

class HealthMetric(BaseModel):
    metric_name: str
    value: float
    unit: str
    confidence_score: float
    reference_range: Optional[str] = None
    is_abnormal: Optional[bool] = None
    source_text: Optional[str] = None

class OCRResponse(BaseModel):
    image_id: str
    status: ProcessingStatus
    extracted_metrics: List[HealthMetric]
    raw_ocr_text: str
    processing_time_seconds: float
    needs_human_validation: bool
    validation_notes: List[str] = []
    preprocessing_applied: bool
    ocr_engine: str
    llm_usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None