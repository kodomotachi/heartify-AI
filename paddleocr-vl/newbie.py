import cv2
import time

from app.services.ocr.vl_engine import vl_engine

img = cv2.imread("test.png")  # ảnh test của anh

print("⏳ Predicting with PaddleOCR-VL...")
start = time.time()

result = vl_engine.predict(
    img,
    use_layout_detection=True,
    use_chart_recognition=False
)

print(f"✅ Done in {time.time() - start:.2f}s")
print(result)
