import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms


def load_image(image_path: str, mode: str = "RGB") -> Image.Image:
    """
    Load an image from disk as a PIL Image.

    Args:
        image_path: Path to the image file.
        mode: PIL mode ('RGB', 'RGBA', 'L', etc.).

    Returns:
        PIL Image object.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert(mode)


def save_image(image, output_path: str) -> str:
    """
    Save a PIL Image or numpy array to disk.

    Args:
        image: PIL Image or numpy ndarray.
        output_path: Destination file path.

    Returns:
        Path to saved file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(output_path)
    return output_path


def resize_image(image: Image.Image, size: tuple) -> Image.Image:
    """
    Resize a PIL Image to the given (width, height).

    Args:
        image: Input PIL Image.
        size: Target (width, height) tuple.

    Returns:
        Resized PIL Image.
    """
    return image.resize(size, Image.LANCZOS)


def normalize_image(image: np.ndarray, mean: list = None, std: list = None) -> np.ndarray:
    """
    Normalize a numpy image array to zero mean and unit std.

    Args:
        image: HxWxC float32 array in [0, 1].
        mean: Per-channel mean list. Defaults to [0.5, 0.5, 0.5].
        std: Per-channel std list. Defaults to [0.5, 0.5, 0.5].

    Returns:
        Normalized float32 numpy array.
    """
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]
    return (image - np.array(mean)) / np.array(std)


def image_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """
    Convert a PIL Image to a normalized PyTorch tensor.

    Args:
        image: PIL Image.
        normalize: Whether to normalize to [-1, 1].

    Returns:
        Tensor of shape (1, C, H, W).
    """
    t = [transforms.ToTensor()]
    if normalize:
        t.append(transforms.Normalize(mean=[0.5]*3, std=[0.5]*3))
    return transforms.Compose(t)(image).unsqueeze(0)


def apply_clahe(image_path: str) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    Improves local contrast, especially useful for OCR preprocessing.

    Args:
        image_path: Path to input image.

    Returns:
        CLAHE-enhanced grayscale image as uint8 numpy array.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def detect_edges_sobel(image: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection to a grayscale image.
    Computes gradient magnitude G = sqrt(Gx^2 + Gy^2).

    Args:
        image: Grayscale uint8 numpy array.

    Returns:
        Edge magnitude map as float32 array.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude.astype(np.float32)
