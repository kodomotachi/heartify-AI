import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "fastapi_ai"))

from app.services.ocr.vl_engine import vl_engine


img = np.ones((500, 500, 3), dtype=np.uint8) * 255
cv2.putText(img, "Blood Pressure: 120/80 mmHg", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(img, "Glucose: 95 mg/dL", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

print("Starting prediction...")
start = time.time()

result = vl_engine.predict(img)

elapsed = time.time() - start
print(f"Done in {elapsed:.2f}s")
print(f"Result: {result}")
