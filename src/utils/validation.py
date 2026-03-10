import os
from pathlib import Path

SUPPORTED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
SUPPORTED_MODEL_EXTS = {'.pth', '.pt', '.pkl', '.h5', '.ckpt'}


def validate_image_path(path: str) -> bool:
    """
    Validate that a file path points to an existing, supported image file.

    Args:
        path: File path to validate.

    Returns:
        True if valid, raises ValueError/FileNotFoundError otherwise.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    if Path(path).suffix.lower() not in SUPPORTED_IMAGE_EXTS:
        raise ValueError(f"Unsupported image format: {Path(path).suffix}. "
                         f"Supported: {SUPPORTED_IMAGE_EXTS}")
    return True


def validate_model_path(path: str) -> bool:
    """
    Validate that a file path points to an existing model checkpoint.

    Args:
        path: File path to validate.

    Returns:
        True if valid, raises ValueError/FileNotFoundError otherwise.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    if Path(path).suffix.lower() not in SUPPORTED_MODEL_EXTS:
        raise ValueError(f"Unsupported model format: {Path(path).suffix}. "
                         f"Supported: {SUPPORTED_MODEL_EXTS}")
    return True


def validate_output_dir(path: str) -> str:
    """
    Validate and create an output directory if it doesn't exist.

    Args:
        path: Directory path.

    Returns:
        Resolved absolute path.
    """
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)
