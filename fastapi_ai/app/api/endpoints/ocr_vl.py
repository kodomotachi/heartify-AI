from urllib import request
from fastapi import APIRouter, HTTPException, BackgroundTasks
from numpy import rint
from app.models.requests import OCRRequest
from app.models.responses import OCRResponse, ProcessingStatus, HealthMetric
from app.services.ocr.vl_engine import vl_engine
from app.services.llm.enhancer import medical_enhancer
from app.utils.image_untils import download_image_fast
from app.utils.callbacks import send_callback_to_nestjs
from app.config import settings
import logging
import time

router = APIRouter(prefix="/api/ocr", tags=["OCR"])
logger = logging.getLogger(__name__)


@router.post("/extract-metrics", response_model=OCRResponse)
async def extract_health_metrics(
    request: OCRRequest,
    background_tasks: BackgroundTasks
):
    """
    Production OCR endpoint with PaddleOCR-VL
    """
    overall_start = time.time()
    
    try:
        # Download image
        start = time.time()
        image, _ = await download_image_fast(request.image_url)
        download_time = time.time() - start
        
        # VL Prediction
        start = time.time()
        use_layout = request.use_layout_detection if request.use_layout_detection is not None else settings.USE_LAYOUT_DETECTION
        use_chart = request.use_chart_recognition if request.use_chart_recognition is not None else settings.USE_CHART_RECOGNITION
        
        vl_results = vl_engine.predict(
            image,
            use_layout_detection=use_layout,
            use_chart_recognition=use_chart,
        )
        vl_time = time.time() - start
        
        # Get first page result
        page_result = vl_results[0]
        json_result = page_result["json"]
        markdown_result = page_result["markdown"]
  
        res_data = json_result.get("res", {})
        
        # Optional LLM Enhancement
        llm_time = 0
        use_llm = request.use_llm_enhancement if request.use_llm_enhancement is not None else settings.USE_LLM_ENHANCEMENT
       
        print(f"🔍 LLM CHECK: request.use_llm_enhancement={request.use_llm_enhancement}, settings.USE_LLM_ENHANCEMENT={settings.USE_LLM_ENHANCEMENT}, final use_llm={use_llm}")
        if use_llm:
            start = time.time()
           
            enhanced_data = medical_enhancer.enhance({"parsing_res_list": res_data.get("parsing_res_list", [])})
            llm_time = time.time() - start
            # Get enhanced metrics
            metrics = enhanced_data.get("enhanced_metrics", [])
        else:
            metrics = []
        
     
        parsing_results = res_data.get("parsing_res_list", [])
        
        if not metrics:
           
            for block in parsing_results:
                if block["block_label"] in ["table", "text"]:
                    # TODO:
                   
                    pass
        
        # Build response
        total_time = time.time() - overall_start
        
        response = OCRResponse(
            image_id=request.image_id,
            status=ProcessingStatus.COMPLETED if metrics else ProcessingStatus.NEEDS_VALIDATION,
            extracted_metrics=[HealthMetric(**m) for m in metrics],
            raw_ocr_text=markdown_result.get("markdown_texts", "")[:1000] if markdown_result else "",
            processing_time_seconds=total_time,
            needs_human_validation=len(metrics) == 0,
            validation_notes=[] if metrics else ["No metrics extracted - manual review needed"],
            preprocessing_applied=False,
            ocr_engine=f"paddleocr_vl_{settings.VL_BACKEND}",
            llm_usage=enhanced_data.get("llm_usage") if use_llm else None,
            metadata={
                "backend": settings.VL_BACKEND,
                "layout_detection": use_layout,
                "chart_recognition": use_chart,
                "blocks_parsed": len(parsing_results),
                "timing": {
                    "download": round(download_time, 2),
                    "vl_prediction": round(vl_time, 2),
                    "llm_enhancement": round(llm_time, 2),
                }
            }
        )
        
        # Async callback
        if settings.ENABLE_BACKGROUND_PROCESSING:
            callback_url = request.callback_url or f"{settings.NESTJS_CALLBACK_URL}/api/webhooks/ocr-complete"
            background_tasks.add_task(
                send_callback_to_nestjs,
                str(callback_url),
                request.image_id,
                request.user_id,
                response.dict()
            )
        
        logger.info(f"✓ Request {request.image_id}: {total_time:.2f}s, {len(parsing_results)} blocks parsed, {len(metrics)} metrics extracted")
        
        return response
        
    except Exception as e:
        logger.error(f"Request {request.image_id} failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Processing failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine": "PaddleOCR-VL-0.9B",
        "backend": settings.VL_BACKEND,
        "server_url": settings.VL_SERVER_URL,
        "features": {
            "layout_detection": settings.USE_LAYOUT_DETECTION,
            "chart_recognition": settings.USE_CHART_RECOGNITION,
            "llm_enhancement": settings.USE_LLM_ENHANCEMENT,
        }
    }
