from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

img_path = "test.png"

result = ocr.ocr(img_path)

for line in result:
    print(line)
