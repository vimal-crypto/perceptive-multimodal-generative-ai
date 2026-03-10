import os
import shutil
from pathlib import Path

SUPPORTED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def ensure_dir(path: str) -> str:
    """
    Create a directory (and all parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path string.
    """
    os.makedirs(path, exist_ok=True)
    return path


def list_image_files(folder: str) -> list:
    """
    List all supported image files in a directory (non-recursive).

    Args:
        folder: Path to the directory.

    Returns:
        Sorted list of full file paths.
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Not a directory: {folder}")
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTS
    ])


def get_output_path(input_path: str, output_dir: str, suffix: str = "", ext: str = None) -> str:
    """
    Construct an output file path based on an input file's stem.

    Args:
        input_path: Path to the input file.
        output_dir: Directory for the output file.
        suffix: Optional suffix to append to the stem (e.g. '_depth').
        ext: Output file extension (e.g. '.png'). Defaults to input extension.

    Returns:
        Full output file path string.
    """
    stem = Path(input_path).stem
    if ext is None:
        ext = Path(input_path).suffix
    ensure_dir(output_dir)
    return os.path.join(output_dir, f"{stem}{suffix}{ext}")


def copy_file(src: str, dst: str) -> str:
    """
    Copy a file from src to dst, creating dst directories if needed.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        Destination path.
    """
    os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else ".", exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def clean_outputs(output_dir: str):
    """
    Delete all files in the outputs directory (non-recursive).

    Args:
        output_dir: Directory to clean.
    """
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            fp = os.path.join(output_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
        print(f"[INFO] Cleaned outputs in: {output_dir}")
