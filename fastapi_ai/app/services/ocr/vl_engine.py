from paddleocr import PaddleOCRVL
from typing import List, Dict, Any, Optional
import logging
import time
from app.config import settings

logger = logging.getLogger(__name__)

class VLEngine:
    """Production PaddleOCR-VL Engine Wrapper"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        logger.info(" Initializing PaddleOCR-VL Engine...")
        
        # Build init params
        init_params = {
            "vl_rec_backend": settings.VL_BACKEND,
            "vl_rec_server_url": settings.VL_SERVER_URL,
            "vl_rec_max_concurrency": settings.VL_MAX_CONCURRENCY,
            "use_layout_detection": settings.USE_LAYOUT_DETECTION,
            "use_doc_orientation_classify": settings.USE_DOC_ORIENTATION_CLASSIFY,
            "use_doc_unwarping": settings.USE_DOC_UNWARPING,
            "use_chart_recognition": settings.USE_CHART_RECOGNITION,
            "format_block_content": settings.FORMAT_BLOCK_CONTENT,
            "layout_threshold": settings.LAYOUT_THRESHOLD,
            "layout_nms": settings.LAYOUT_NMS,
            "layout_unclip_ratio": settings.LAYOUT_UNCLIP_RATIO,
            "layout_merge_bboxes_mode": settings.LAYOUT_MERGE_BBOXES_MODE,
            "device": settings.DEVICE,
        }
        
        # Add optional params
        if settings.VL_MODEL_NAME:
            init_params["vl_rec_model_name"] = settings.VL_MODEL_NAME
        if settings.VL_MODEL_DIR:
            init_params["vl_rec_model_dir"] = settings.VL_MODEL_DIR
        if settings.VL_API_KEY:
            init_params["vl_rec_api_key"] = settings.VL_API_KEY
        if settings.LAYOUT_MODEL_NAME:
            init_params["layout_detection_model_name"] = settings.LAYOUT_MODEL_NAME
        if settings.LAYOUT_MODEL_DIR:
            init_params["layout_detection_model_dir"] = settings.LAYOUT_MODEL_DIR
        
        try:
            self.pipeline = PaddleOCRVL(**init_params)
            self._initialized = True
            logger.info(f" VL Engine ready (backend={settings.VL_BACKEND})")
        except Exception as e:
            logger.error(f"Failed to initialize VL Engine: {e}")
            raise
    
 

    def predict(self, image_input: Any, **kwargs) -> List[Dict[str, Any]]:
            """Run VL prediction"""
            start = time.time()

            predict_params = {
                "use_layout_detection": kwargs.get("use_layout_detection", settings.USE_LAYOUT_DETECTION),
                "use_chart_recognition": kwargs.get("use_chart_recognition", settings.USE_CHART_RECOGNITION),
                "use_queues": settings.USE_QUEUES,
                "repetition_penalty": settings.VL_REPETITION_PENALTY,
                "temperature": settings.VL_TEMPERATURE,
                "top_p": settings.VL_TOP_P,
                "min_pixels": settings.VL_MIN_PIXELS,
                "max_pixels": settings.VL_MAX_PIXELS,
                "format_block_content": settings.FORMAT_BLOCK_CONTENT,
                # "num_workers": 0
            }

            try:
                results = self.pipeline.predict(
                    input=image_input,
                    task_type="document",      
                    use_layout_detection=True,
                    format_block_content=True, 
                    #  use_parallel=False
                )

              
                result_list = list(results)
                logger.info(f"🔍 VL returned {len(result_list)} result(s)")

                for idx, res in enumerate(result_list):
                    logger.info(f"=== RESULT {idx} ===")
                    logger.info(f"Type: {type(res)}")
                    logger.info(f"Dir: {dir(res)}")

                    # In JSON
                    if hasattr(res, 'json'):
                        logger.info(f"JSON keys: {res.json.keys() if res.json else 'None'}")
                        logger.info(f"JSON: {res.json}")

                        # Kiểm tra parsing_res_list
                        if res.json and 'parsing_res_list' in res.json:
                            blocks = res.json['parsing_res_list']
                            logger.info(f" Found {len(blocks)} blocks")
                            for i, block in enumerate(blocks[:3]):  # In 3 block đầu
                                logger.info(f"Block {i}: {block.get('block_label')} - {block.get('block_content', '')[:100]}")

                    # In Markdown
                    if hasattr(res, 'markdown'):
                        logger.info(f"Markdown keys: {res.markdown.keys() if res.markdown else 'None'}")
                        md_text = res.markdown.get('markdown_texts', '') if res.markdown else ''
                        logger.info(f"Markdown text (first 200 chars): {md_text[:200]}")

                # ============================================
                # Convert to dict
                # ============================================
                output = []
                for res in result_list:
                    output.append({
                        "json": res.json if hasattr(res, 'json') else {},
                        "markdown": res.markdown if hasattr(res, 'markdown') else {},
                    })

                processing_time = time.time() - start
                logger.info(f" VL Prediction: {len(output)} page(s) in {processing_time:.2f}s")
                # logger.error(type(res.json["parsing_res_list"][0]["confidence"]))


                return output

            except Exception as e:
                logger.error(f"VL prediction failed: {e}", exc_info=True)
                raise

        # Singleton instance
vl_engine = VLEngine()