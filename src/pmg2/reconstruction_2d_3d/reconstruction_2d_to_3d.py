"""
PMG-2: Complete 2D to 3D Reconstruction Pipeline

Integrated pipeline combining depth estimation, point cloud generation,
and mesh reconstruction for end-to-end 2D to 3D conversion.

Author: PMG-AI Team
Date: 2024
"""

import os
import cv2
import numpy as np
import torch
import open3d as o3d
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
from pathlib import Path


class Complete2Dto3DPipeline:
    """
    End-to-end pipeline for converting 2D images to 3D models.
    """
    
    def __init__(self, 
                 model_type: str = 'DPT_Large',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the reconstruction pipeline."""
        self.device = device
        self.model_type = model_type
        self.midas = None
        self.transform = None
        print(f"Initializing pipeline on {self.device}...")
        self._load_depth_model()
    
    def _load_depth_model(self) -> None:
        """Load MiDaS depth estimation model."""
        print(f"Loading {self.model_type} model...")
        self.midas = torch.hub.load('intel-isl/MiDaS', self.model_type)
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        if self.model_type in ['DPT_Large', 'DPT_Hybrid']:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        print("Depth model loaded successfully!")
    
    def estimate_depth(self, image: np.ndarray, output_path: Optional[str] = None) -> np.ndarray:
        """Estimate depth map from RGB image."""
        print("Estimating depth...")
        input_batch = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        if output_path:
            self._save_depth_visualization(depth_map, image, output_path)
        return depth_map
    
    def _save_depth_visualization(self, depth_map: np.ndarray, rgb_image: np.ndarray, output_path: str) -> None:
        """Save depth map visualization."""
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(rgb_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        im = axes[1].imshow(depth_map, cmap='magma')
        axes[1].set_title('Depth Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_point_cloud(self, depth_map: np.ndarray, rgb_image: np.ndarray, 
                            focal_length: float = 1000.0) -> o3d.geometry.PointCloud:
        """Generate 3D point cloud from depth map."""
        print("Generating point cloud...")
        height, width = depth_map.shape
        cx, cy = width / 2, height / 2
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_map.copy()
        x = (u - cx) * z / focal_length
        y = (v - cy) * z / focal_length
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb_image.reshape(-1, 3) / 255.0
        valid_mask = points[:, 2] > 0
        points = points[valid_mask]
        colors = colors[valid_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"Point cloud generated with {len(pcd.points)} points")
        return pcd
    
    def filter_point_cloud(self, point_cloud: o3d.geometry.PointCloud, 
                          nb_neighbors: int = 20, std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
        """Filter outliers from point cloud."""
        print("Filtering outliers...")
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return point_cloud.select_by_index(ind)
    
    def reconstruct_mesh(self, point_cloud: o3d.geometry.PointCloud, depth: int = 9) -> o3d.geometry.TriangleMesh:
        """Reconstruct 3D mesh from point cloud."""
        print("Reconstructing mesh...")
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        point_cloud.orient_normals_consistent_tangent_plane(k=10)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh
    
    def process_image(self, image_path: str, output_dir: str = './outputs') -> Dict:
        """Process image through complete pipeline."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing: {image_path}")
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth_map = self.estimate_depth(img_rgb, str(output_path / 'depth_output.jpg'))
        point_cloud = self.generate_point_cloud(depth_map, img_rgb)
        point_cloud_filtered = self.filter_point_cloud(point_cloud)
        o3d.io.write_point_cloud(str(output_path / 'point_cloud.ply'), point_cloud_filtered)
        mesh = self.reconstruct_mesh(point_cloud_filtered)
        o3d.io.write_triangle_mesh(str(output_path / 'reconstructed_mesh.ply'), mesh)
        return {'rgb_image': img_rgb, 'depth_map': depth_map, 'point_cloud': point_cloud_filtered, 'mesh': mesh}


def main():
    """Example usage."""
    pipeline = Complete2Dto3DPipeline(model_type='DPT_Large')
    results = pipeline.process_image('input_image.jpg', './outputs')
    o3d.visualization.draw_geometries([results['point_cloud']], window_name='Point Cloud')
    o3d.visualization.draw_geometries([results['mesh']], window_name='Mesh')


if __name__ == '__main__':
    main()
