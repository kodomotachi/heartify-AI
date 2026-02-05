from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from  app.api.endpoints import ocr_vl
from app.config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Heartify AI - Production OCR",
    version="1.0.0-production",
    description="PaddleOCR-VL with vLLM acceleration for medical documents"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr_vl.router)

@app.get("/")
async def root():
    return {
        "message": "Heartify AI Production API",
        "engine": "PaddleOCR-VL-0.9B",
        "backend": settings.VL_BACKEND,
        "docs": "/docs"
    }

@app.get("/health")
async def main_health():
    return {
        "status": "healthy",
        "backend": settings.VL_BACKEND,
        "vllm_server": settings.VL_SERVER_URL,
    }
