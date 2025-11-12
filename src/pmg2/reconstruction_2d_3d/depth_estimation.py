"""
MiDaS Depth Estimation Module for PMG-2

This module provides depth estimation functionality using the MiDaS model
for 2D-to-3D reconstruction tasks.

Based on: Intel ISL MiDaS (Mixed Data Sampling for Depth Estimation)
Author: PMG-AI Team
Date: 2024
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import os
from typing import Optional, Tuple, Union
import logging
from torchvision.transforms import Compose, Normalize, ToTensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MiDaSDepthEstimator:
    """
    Depth estimation using MiDaS (Mixed Data Sampling) model.
    
    This class implements monocular depth estimation for 2D-to-3D reconstruction,
    supporting multiple MiDaS model variants with different accuracy/speed tradeoffs.
    
    Attributes:
        model_type (str): Type of MiDaS model ('large', 'small', 'hybrid')
        device (str): Compute device ('cuda' or 'cpu')
        model: Loaded MiDaS model
        transform: Image preprocessing transform
    """
    
    def __init__(
        self,
        model_type: str = "small",
        device: str = "cuda"
    ):
        """
        Initialize the MiDaS depth estimator.
        
        Args:
            model_type: Model variant - 'large' (DPT-Large), 'small' (MiDaS_small),
                       or 'hybrid' (DPT-Hybrid)
            device: Compute device ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing MiDaS depth estimator ({model_type}) on {self.device}")
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the MiDaS model from torch hub.
        """
        try:
            # Load appropriate model based on type
            if self.model_type == "large":
                self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            elif self.model_type == "hybrid":
                self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            else:  # small
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if self.model_type in ["large", "hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            logger.info("MiDaS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise
    
    def estimate_depth(
        self,
        image: Union[str, np.ndarray, Image.Image],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Estimate depth map from input image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            normalize: Whether to normalize depth map to [0, 255] range
        
        Returns:
            Depth map as numpy array (height, width)
        """
        # Load and preprocess image
        img_array = self._load_image(image)
        original_shape = img_array.shape[:2]
        
        logger.info(f"Estimating depth for image shape: {original_shape}")
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Apply transforms and prepare input
        input_batch = self.transform(img_array).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=original_shape,
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        
        # Normalize if requested
        if normalize:
            depth_map = self._normalize_depth(depth_map)
        
        logger.info("Depth estimation complete")
        return depth_map
    
    def _load_image(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        Load image from various input types.
        
        Args:
            image: Input image (path, array, or PIL Image)
        
        Returns:
            Image as numpy array
        """
        if isinstance(image, str):
            # Load from file path
            img_array = cv2.imread(image)
            if img_array is None:
                raise ValueError(f"Could not load image from {image}")
        elif isinstance(image, Image.Image):
            # Convert PIL Image to numpy
            img_array = np.array(image)
        elif isinstance(image, np.ndarray):
            img_array = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        return img_array
    
    def _normalize_depth(
        self,
        depth_map: np.ndarray,
        min_val: float = 0,
        max_val: float = 255
    ) -> np.ndarray:
        """
        Normalize depth map to specified range.
        
        Args:
            depth_map: Raw depth map
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization
        
        Returns:
            Normalized depth map
        """
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        normalized = (depth_map - depth_min) / (depth_max - depth_min)
        normalized = normalized * (max_val - min_val) + min_val
        
        return normalized.astype(np.uint8)
    
    def save_depth_map(
        self,
        depth_map: np.ndarray,
        output_path: str,
        colormap: int = cv2.COLORMAP_INFERNO
    ) -> None:
        """
        Save depth map to file with optional colormap.
        
        Args:
            depth_map: Depth map array
            output_path: Path to save the depth map
            colormap: OpenCV colormap (default: COLORMAP_INFERNO)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Apply colormap if not already 3-channel
        if len(depth_map.shape) == 2:
            if colormap is not None:
                depth_colored = cv2.applyColorMap(depth_map, colormap)
            else:
                depth_colored = depth_map
        else:
            depth_colored = depth_map
        
        cv2.imwrite(output_path, depth_colored)
        logger.info(f"Depth map saved to {output_path}")
    
    def estimate_depth_batch(
        self,
        images: list,
        normalize: bool = True
    ) -> list:
        """
        Estimate depth for multiple images.
        
        Args:
            images: List of images (paths, arrays, or PIL Images)
            normalize: Whether to normalize depth maps
        
        Returns:
            List of depth map arrays
        """
        depth_maps = []
        
        logger.info(f"Processing batch of {len(images)} images")
        
        for i, image in enumerate(images):
            depth_map = self.estimate_depth(image, normalize=normalize)
            depth_maps.append(depth_map)
            logger.info(f"Processed image {i+1}/{len(images)}")
        
        return depth_maps
    
    def create_point_cloud(
        self,
        image: Union[str, np.ndarray],
        depth_map: np.ndarray,
        focal_length: Optional[float] = None
    ) -> np.ndarray:
        """
        Create 3D point cloud from image and depth map.
        
        Args:
            image: RGB image
            depth_map: Corresponding depth map
            focal_length: Camera focal length (estimated if None)
        
        Returns:
            Point cloud as (N, 6) array [x, y, z, r, g, b]
        """
        # Load image if path provided
        img_array = self._load_image(image)
        if len(img_array.shape) == 3:
            rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            rgb = img_array
        
        height, width = depth_map.shape
        
        # Estimate focal length if not provided
        if focal_length is None:
            focal_length = width  # Simple estimation
        
        # Create mesh grid
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Calculate 3D coordinates
        x3d = (x - width / 2) * depth_map / focal_length
        y3d = (y - height / 2) * depth_map / focal_length
        z3d = depth_map
        
        # Flatten arrays
        points_3d = np.stack([x3d, y3d, z3d], axis=-1).reshape(-1, 3)
        colors = rgb.reshape(-1, 3)
        
        # Combine position and color
        point_cloud = np.hstack([points_3d, colors])
        
        logger.info(f"Generated point cloud with {len(point_cloud)} points")
        return point_cloud


def main():
    """
    Example usage of the MiDaS depth estimator.
    """
    # Initialize estimator
    estimator = MiDaSDepthEstimator(model_type="small", device="cuda")
    
    # Estimate depth from image
    image_path = "input/car.jpg"  # Change to your image path
    
    if os.path.exists(image_path):
        # Estimate depth
        depth_map = estimator.estimate_depth(image_path, normalize=True)
        
        # Save depth map
        estimator.save_depth_map(
            depth_map,
            "outputs/depth/car_depth_output.jpg",
            colormap=cv2.COLORMAP_INFERNO
        )
        
        # Create point cloud
        point_cloud = estimator.create_point_cloud(image_path, depth_map)
        
        # Save point cloud (simple format)
        np.savetxt(
            "outputs/depth/car_point_cloud.xyz",
            point_cloud,
            fmt='%.6f',
            header='x y z r g b',
            comments=''
        )
        
        print("Depth estimation complete!")
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")


if __name__ == "__main__":
    main()
