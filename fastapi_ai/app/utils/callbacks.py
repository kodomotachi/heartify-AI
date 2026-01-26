import httpx
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def send_callback_to_nestjs(
    callback_url: str,
    image_id: str,
    user_id: str,
    result_data: Dict[str, Any]
):
    """Send async callback to NestJS"""
    try:
        payload = {
            "image_id": image_id,
            "user_id": user_id,
            "result": result_data,
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()
            logger.info(f"✓ Callback sent: {image_id}")
            
    except Exception as e:
        logger.error(f"Callback failed for {image_id}: {e}")
