import os
import torch
import numpy as np
from PIL import Image
import cv2


class MiDaSDepthEstimator:
    """
    Monocular depth estimator using Intel MiDaS (DPT_Large).
    Estimates per-pixel depth from a single RGB image.

    Reference: Ranftl et al., "Towards Robust Monocular Depth Estimation:
    Mixing Datasets for Zero-shot Cross-dataset Transfer", TPAMI 2022.
    """

    def __init__(self, model_type: str = "DPT_Large", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[INFO] Loading MiDaS model: {model_type}")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.eval()
        self.model.to(device)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform
        print("[INFO] MiDaS model loaded.")

    def estimate(self, image_path: str) -> np.ndarray:
        """
        Estimate depth map from an input image.

        Args:
            image_path: Path to the RGB input image.

        Returns:
            Depth map as a 2D float32 numpy array, normalized to [0, 1].
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)

    def save_depth_map(self, depth: np.ndarray, output_path: str) -> str:
        """
        Save a depth map as a grayscale PNG.

        Args:
            depth: Normalized depth array [0, 1].
            output_path: Path to save the depth image.

        Returns:
            Path to saved file.
        """
        depth_img = (depth * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, depth_img)
        print(f"[INFO] Depth map saved: {output_path}")
        return output_path


def estimate_depth(image_path: str, model_type: str = "DPT_Large", device: str = None) -> np.ndarray:
    """
    Convenience function: load MiDaS and estimate depth for a single image.

    Args:
        image_path: Path to input image.
        model_type: MiDaS model variant.
        device: Compute device.

    Returns:
        Normalized depth map as float32 numpy array.
    """
    estimator = MiDaSDepthEstimator(model_type=model_type, device=device)
    return estimator.estimate(image_path)
