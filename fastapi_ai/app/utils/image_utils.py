import httpx
import cv2
import numpy as np
from typing import Tuple

async def download_image_fast(url: str) -> Tuple[np.ndarray, bytes]:
    """Download and decode image"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(str(url))
        response.raise_for_status()
        
        image_bytes = response.content
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image, image_bytes
