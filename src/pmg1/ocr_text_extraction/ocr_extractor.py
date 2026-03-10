import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# Configure Tesseract path for Windows if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_for_ocr(image_path: str) -> np.ndarray:
    """
    Preprocess an image to improve OCR accuracy.
    Applies grayscale conversion, adaptive histogram equalization,
    Gaussian denoising, and binary thresholding.

    Args:
        image_path: Path to the input image.

    Returns:
        Preprocessed image as a numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu's binarization
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological dilation to connect broken text characters
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)

    return processed


def extract_text_from_image(
    image_path: str,
    lang: str = 'eng',
    preprocess: bool = True,
    config: str = '--oem 3 --psm 6'
) -> str:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        image_path: Path to the input image file.
        lang: Tesseract language code (default 'eng').
        preprocess: Whether to apply preprocessing before OCR.
        config: Tesseract configuration string.

    Returns:
        Extracted text as a string.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if preprocess:
        processed = preprocess_for_ocr(image_path)
        pil_image = Image.fromarray(processed)
    else:
        pil_image = Image.open(image_path)

    text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
    return text.strip()


def extract_text_with_bboxes(
    image_path: str,
    lang: str = 'eng'
) -> list:
    """
    Extract text along with bounding box coordinates for each word.

    Args:
        image_path: Path to the input image.
        lang: Tesseract language code.

    Returns:
        List of dicts with keys: 'text', 'left', 'top', 'width', 'height', 'conf'.
    """
    pil_image = Image.open(image_path)
    data = pytesseract.image_to_data(pil_image, lang=lang, output_type=pytesseract.Output.DICT)

    results = []
    for i, word in enumerate(data['text']):
        if word.strip() and int(data['conf'][i]) > 30:
            results.append({
                'text': word,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'conf': data['conf'][i]
            })
    return results


def save_extracted_text(text: str, output_path: str) -> str:
    """
    Save extracted OCR text to a .txt file.

    Args:
        text: The extracted text string.
        output_path: File path to save to.

    Returns:
        Path to the saved file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"[INFO] Text saved to: {output_path}")
    return output_path
