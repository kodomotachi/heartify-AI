import google.generativeai as genai
import json
import logging
from typing import Dict, Any, List
from app.config import settings

logger = logging.getLogger(__name__)

class MedicalEnhancer:
    """Optional Gemini enhancement for medical data"""
    
    def __init__(self):
        self.model = None
        if settings.USE_LLM_ENHANCEMENT and settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
    
    def enhance(self, vl_result: Dict) -> Dict:
        """Enhance VL results with structured medical extraction"""
        if not self.model:
            return vl_result
        
        # Extract text
        parsing_results = vl_result.get("parsing_res_list", [])
        text_blocks = [
            f"[{b['block_label']}] {b['block_content']}"
            for b in parsing_results[:10]  # Limit for token efficiency
        ]
        full_text = "\n\n".join(text_blocks)
        
        prompt = f"""Extract health metrics from this medical report.
TEXT:
{full_text[:3000]}
Return JSON with metrics:
{{
  "metrics": [
    {{"metric_name": "...", "value": 0, "unit": "...", "confidence_score": 0.9, "reference_range": "..."}}
  ]
}}
Focus on: BP, Glucose, Cholesterol, Creatinine. Return ONLY JSON."""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=settings.GEMINI_TEMPERATURE,
                    max_output_tokens=settings.GEMINI_MAX_TOKENS,
                )
            )
            
            response_text = response.text.strip()
            if "```" in response_text:
                response_text = response_text.split("```")[1].replace("json", "").strip()
            
            enhanced = json.loads(response_text)
            vl_result["enhanced_metrics"] = enhanced.get("metrics", [])
            vl_result["llm_usage"] = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }
            
            return vl_result
            
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return vl_result

medical_enhancer = MedicalEnhancer()