# test_predict.py
import cv2
import numpy as np
import time
from app.services.ocr.vl_engine import vl_engine

# Tạo image test đơn giản
img = np.ones((500, 500, 3), dtype=np.uint8) * 255
cv2.putText(img, "Blood Pressure: 120/80 mmHg", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
cv2.putText(img, "Glucose: 95 mg/dL", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

print("Starting prediction...")
start = time.time()

result = vl_engine.predict(img)  # Gọi trực tiếp, không qua FastAPI

elapsed = time.time() - start
print(f"Done in {elapsed:.2f}s")
print(f"Result: {result}")