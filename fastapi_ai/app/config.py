import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # ========== API Keys ==========
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # ========== Gemini Settings ==========
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_TOKENS: int = 1024
    
    # ========== NestJS Integration ==========
    NESTJS_CALLBACK_URL: str = "http://localhost:3000"
    VL_SERVER_URL: Optional[str] = None
    
    # ========== VL Backend Configuration ==========
    VL_BACKEND: str = "native"  # native | vllm-server | sglang-server | fastdeploy-server
    VL_API_KEY: Optional[str] = None
    VL_MAX_CONCURRENCY: int = 4
    
    # ========== Model Configuration ==========
    VL_MODEL_NAME: str = "PaddleOCR-VL-0.9B"    
    VL_MODEL_DIR: Optional[str] = None
    
    # ========== VL Inference Settings ==========
    VL_TEMPERATURE: float = 0.0
    VL_TOP_P: float = 0.9
    VL_REPETITION_PENALTY: float = 1.1
    VL_MIN_PIXELS: int = 256 * 28 * 28
    VL_MAX_PIXELS: int = 1024 * 28 * 28
    
    # ========== Document Processing ==========
    USE_LAYOUT_DETECTION: bool = True
    USE_DOC_ORIENTATION_CLASSIFY: bool = False
    USE_DOC_UNWARPING: bool = False
    USE_CHART_RECOGNITION: bool = False
    FORMAT_BLOCK_CONTENT: bool = True
    USE_QUEUES: bool = True
    
    # ========== Layout Detection Config ==========
    LAYOUT_MODEL_NAME: Optional[str] = None
    LAYOUT_MODEL_DIR: Optional[str] = None
    LAYOUT_THRESHOLD: float = 0.5
    LAYOUT_NMS: bool = True
    LAYOUT_UNCLIP_RATIO: float = 1.5
    LAYOUT_MERGE_BBOXES_MODE: str = "union"
    
    # ========== Output Settings ==========
    PRETTIFY_MARKDOWN: bool = True
    SHOW_FORMULA_NUMBER: bool = False
    
    # ========== LLM Enhancement ==========
    USE_LLM_ENHANCEMENT: bool = True
    
    # ========== Performance Settings ==========
    ENABLE_BACKGROUND_PROCESSING: bool = True
    CALLBACK_RETRY_MAX: int = 2
    MAX_IMAGE_SIZE_MB: int = 10
    TIMEOUT_SECONDS: int = 60
    
    # ========== Device ==========
    DEVICE: str = "gpu:0"  # cpu, gpu:0 
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()