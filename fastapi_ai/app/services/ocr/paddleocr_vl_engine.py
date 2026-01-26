from paddleocr import PaddleOCRVL
from typing import Dict, List, Any, Optional
import logging
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)


class PaddleOCRVLEngine:
 
    
    def __init__(self):
        logger.info(" Initializing PaddleOCR-VL Official Engine...")
        
        # ============================================
        # Initialize PaddleOCR-VL Pipeline
        # ============================================
        init_params = {
            # Device
            "device": settings.PADDLE_DEVICE,
            
            # VL Model
            "vl_rec_model_name": settings.PADDLE_VL_MODEL_NAME,
            "vl_rec_model_dir": settings.PADDLE_VL_MODEL_DIR,
            "vl_rec_backend": settings.PADDLE_VL_BACKEND,
            
            # Layout Detection
            "use_layout_detection": settings.PADDLE_USE_LAYOUT_DETECTION,
            "layout_threshold": settings.PADDLE_LAYOUT_THRESHOLD,
            "layout_nms": settings.PADDLE_LAYOUT_NMS,
            "layout_unclip_ratio": settings.PADDLE_LAYOUT_UNCLIP_RATIO,
            
            # Document Preprocessing
            "use_doc_orientation_classify": settings.PADDLE_USE_DOC_ORIENTATION,
            "use_doc_unwarping": settings.PADDLE_USE_DOC_UNWARPING,
            
            # Chart Recognition
            "use_chart_recognition": settings.PADDLE_USE_CHART_RECOGNITION,
            
            # Output Format
            "format_block_content": settings.PADDLE_FORMAT_BLOCK_CONTENT,
        }
        
        # VL Server configuration (if using vllm/sglang)
        if settings.PADDLE_VL_BACKEND in ["vllm-server", "sglang-server"]:
            if not settings.PADDLE_VL_SERVER_URL:
                raise ValueError(
                    f"VL_SERVER_URL required for backend={settings.PADDLE_VL_BACKEND}"
                )
            init_params["vl_rec_server_url"] = settings.PADDLE_VL_SERVER_URL
            init_params["vl_rec_max_concurrency"] = settings.PADDLE_VL_MAX_CONCURRENCY
            logger.info(
                f"✓ Using {settings.PADDLE_VL_BACKEND} at {settings.PADDLE_VL_SERVER_URL}"
            )
        
        # Initialize pipeline
        self.pipeline = PaddleOCRVL(**init_params)
        
        logger.info(
            f"✓ PaddleOCR-VL ready (backend={settings.PADDLE_VL_BACKEND}, "
            f"device={settings.PADDLE_DEVICE})"
        )
    
    
    def extract_from_image(
        self, 
        image: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract from image using PaddleOCR-VL
        
        Args:
            image: numpy array (BGR format from cv2)
            **kwargs: Override default settings
        
        Returns:
            {
                "parsing_results": [...],  # List of parsed blocks
                "markdown": {...},         # Markdown output
                "processing_time": 2.5,
                "metadata": {...}
            }
        """
        start = time.time()
        
        try:
            # ============================================
            # Predict Parameters
            # ============================================
            predict_params = {
                "use_layout_detection": kwargs.get(
                    "use_layout_detection", 
                    settings.PADDLE_USE_LAYOUT_DETECTION
                ),
                "use_doc_orientation_classify": kwargs.get(
                    "use_doc_orientation_classify",
                    settings.PADDLE_USE_DOC_ORIENTATION
                ),
                "use_doc_unwarping": kwargs.get(
                    "use_doc_unwarping",
                    settings.PADDLE_USE_DOC_UNWARPING
                ),
                "use_chart_recognition": kwargs.get(
                    "use_chart_recognition",
                    settings.PADDLE_USE_CHART_RECOGNITION
                ),
                "format_block_content": kwargs.get(
                    "format_block_content",
                    settings.PADDLE_FORMAT_BLOCK_CONTENT
                ),
                "use_queues": kwargs.get(
                    "use_queues",
                    settings.PADDLE_USE_QUEUES
                ),
                "temperature": kwargs.get(
                    "temperature",
                    settings.PADDLE_VL_TEMPERATURE
                ),
                "top_p": kwargs.get(
                    "top_p",
                    settings.PADDLE_VL_TOP_P
                ),
                "repetition_penalty": kwargs.get(
                    "repetition_penalty",
                    settings.PADDLE_VL_REPETITION_PENALTY
                ),
                "min_pixels": kwargs.get(
                    "min_pixels",
                    settings.PADDLE_VL_MIN_PIXELS
                ),
                "max_pixels": kwargs.get(
                    "max_pixels",
                    settings.PADDLE_VL_MAX_PIXELS
                ),
            }
            
            # ============================================
            # Run Prediction
            # ============================================
            results = self.pipeline.predict(image, **predict_params)
            
            # Process results (generator → list)
            result_list = list(results)
            
            if not result_list:
                raise ValueError("No results from PaddleOCR-VL")
            
            # First result (single image)
            result = result_list[0]
            
            # ============================================
            # Extract Data
            # ============================================
            
            # Get JSON representation
            json_data = result.json
            
            # Get Markdown
            markdown_data = result.markdown
            
            # Parsing results
            parsing_results = json_data.get("parsing_res_list", [])
            
            processing_time = time.time() - start
            
            logger.info(
                f" PaddleOCR-VL: {len(parsing_results)} blocks in {processing_time:.2f}s"
            )
            
            # ============================================
            # Return Structured Data
            # ============================================
            return {
                "parsing_results": parsing_results,
                "markdown": {
                    "text": markdown_data.get("markdown_texts", ""),
                    "images": markdown_data.get("markdown_images", {}),
                    "is_start": markdown_data.get("page_continuation_flags", (True, True))[0],
                    "is_end": markdown_data.get("page_continuation_flags", (True, True))[1],
                },
                "processing_time": processing_time,
                "metadata": {
                    "model_settings": json_data.get("model_settings", {}),
                    "num_blocks": len(parsing_results),
                    "backend": settings.PADDLE_VL_BACKEND,
                },
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR-VL extraction failed: {str(e)}")
            raise
    
    
    def extract_medical_metrics(
        self, 
        parsing_results: List[Dict]
    ) -> List[Dict]:
        """
        Extract medical metrics from parsing results
        
        Args:
            parsing_results: Output from extract_from_image()
        
        Returns:
            List of medical metrics with values
        """
        medical_data = []
        
        for block in parsing_results:
            block_label = block.get("block_label", "")
            block_content = block.get("block_content", "")
            
            # Focus on text and table blocks
            if block_label in ["text", "table", "paragraph_title"]:
                # Simple keyword matching for medical terms
                medical_keywords = {
                    "blood pressure": ["systolic", "diastolic", "bp", "mmhg"],
                    "glucose": ["glucose", "blood sugar", "fasting"],
                    "cholesterol": ["cholesterol", "ldl", "hdl", "triglycerides"],
                    "creatinine": ["creatinine", "kidney"],
                }
                
                content_lower = block_content.lower()
                
                for metric_type, keywords in medical_keywords.items():
                    if any(kw in content_lower for kw in keywords):
                        medical_data.append({
                            "metric_type": metric_type,
                            "content": block_content,
                            "bbox": block.get("block_bbox"),
                            "block_id": block.get("block_id"),
                        })
        
        return medical_data
