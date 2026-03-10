import pytesseract
import cv2
import numpy as np
from PIL import Image

def preprocess_for_ocr(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding for better text visibility
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, h=30)
    return denoised

def extract_text(image_path: str, lang: str = 'eng') -> str:
    preprocessed = preprocess_for_ocr(image_path)
    pil_image = Image.fromarray(preprocessed)
    text = pytesseract.image_to_string(pil_image, lang=lang)
    return text.strip()

def extract_text_with_boxes(image_path: str) -> list:
    preprocessed = preprocess_for_ocr(image_path)
    pil_image = Image.fromarray(preprocessed)
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    results = []
    for i, text in enumerate(data['text']):
        if text.strip():
            results.append({
                'text': text,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'confidence': data['conf'][i]
            })
    return results

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        print(extract_text(sys.argv[1]))
    else:
        print("Usage: python ocr_extractor.py <image_path>")
