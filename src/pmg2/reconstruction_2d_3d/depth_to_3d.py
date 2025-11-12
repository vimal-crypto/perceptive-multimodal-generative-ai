"""
PMG-2: Depth to 3D Reconstruction Module

This module implements Neural Radiance Fields (NeRF) based 3D reconstruction
from depth maps and 2D images.

Author: PMG-AI Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import open3d as o3d


class NeRFModel(nn.Module):
    """
    Neural Radiance Field model for 3D scene representation.
    
    Learns a continuous volumetric scene function that maps 3D coordinates
    and viewing directions to RGB colors and volume density.
    """
    
    def __init__(self, embedding_dim=256, hidden_dim=256, num_layers=8):
        """
        Initialize NeRF model.
        
        Args:
            embedding_dim: Dimension of positional encoding
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super(NeRFModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Input layer for position encoding
        self.fc_in = nn.Linear(embedding_dim * 3, hidden_dim)
        
        # Hidden layers
        self.fc_hidden = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Output layers for RGB and density
        self.fc_density = nn.Linear(hidden_dim, 1)
        self.fc_rgb = nn.Linear(hidden_dim, 3)
        
        # Activation
        self.relu = nn.ReLU()
        
    def positional_encoding(self, x: torch.Tensor, L: int = 10) -> torch.Tensor:
        """
        Apply positional encoding to input coordinates.
        
        Args:
            x: Input coordinates [B, 3]
            L: Number of frequency bands
            
        Returns:
            Encoded coordinates [B, 3 * 2 * L]
        """
        encoding = [x]
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                encoding.append(fn(2.0 ** i * torch.pi * x))
        return torch.cat(encoding, dim=-1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through NeRF network.
        
        Args:
            x: 3D coordinates [B, 3]
            
        Returns:
            rgb: RGB colors [B, 3]
            density: Volume density [B, 1]
        """
        # Positional encoding
        x_encoded = self.positional_encoding(x)
        
        # Forward through network
        h = self.relu(self.fc_in(x_encoded))
        
        for layer in self.fc_hidden:
            h = self.relu(layer(h))
        
        # Output RGB and density
        density = self.fc_density(h)
        rgb = torch.sigmoid(self.fc_rgb(h))
        
        return rgb, density


class DepthTo3DConverter:
    """
    Convert depth maps to 3D point clouds and meshes using neural reconstruction.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize converter.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
        self.nerf_model = None
        
    def depth_to_point_cloud(self,
                             depth_map: np.ndarray,
                             rgb_image: np.ndarray,
                             focal_length: float = 1000.0,
                             principal_point: Optional[Tuple[float, float]] = None) -> o3d.geometry.PointCloud:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            depth_map: Depth map [H, W]
            rgb_image: RGB image [H, W, 3]
            focal_length: Camera focal length
            principal_point: Camera principal point (cx, cy)
            
        Returns:
            Open3D point cloud object
        """
        height, width = depth_map.shape
        
        if principal_point is None:
            cx, cy = width / 2, height / 2
        else:
            cx, cy = principal_point
        
        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Calculate 3D coordinates
        z = depth_map
        x = (u - cx) * z / focal_length
        y = (v - cy) * z / focal_length
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb_image.reshape(-1, 3) / 255.0
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Remove invalid points (zero depth)
        mask = points[:, 2] > 0
        pcd = pcd.select_by_index(np.where(mask)[0])
        
        return pcd
    
    def train_nerf(self,
                   point_cloud: o3d.geometry.PointCloud,
                   num_epochs: int = 1000,
                   learning_rate: float = 1e-4) -> NeRFModel:
        """
        Train NeRF model on point cloud data.
        
        Args:
            point_cloud: Input point cloud
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Trained NeRF model
        """
        # Initialize model
        model = NeRFModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Convert point cloud to tensors
        points = torch.FloatTensor(np.asarray(point_cloud.points)).to(self.device)
        colors = torch.FloatTensor(np.asarray(point_cloud.colors)).to(self.device)
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            # Sample random points
            indices = torch.randint(0, len(points), (min(1024, len(points)),))
            batch_points = points[indices]
            batch_colors = colors[indices]
            
            # Forward pass
            pred_rgb, pred_density = model(batch_points)
            
            # Compute loss
            color_loss = F.mse_loss(pred_rgb, batch_colors)
            density_loss = torch.mean(torch.abs(pred_density))
            loss = color_loss + 0.01 * density_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
        self.nerf_model = model
        return model
    
    def reconstruct_mesh(self,
                         point_cloud: o3d.geometry.PointCloud,
                         method: str = 'poisson',
                         depth: int = 9) -> o3d.geometry.TriangleMesh:
        """
        Reconstruct 3D mesh from point cloud.
        
        Args:
            point_cloud: Input point cloud
            method: Reconstruction method ('poisson' or 'ball_pivoting')
            depth: Octree depth for Poisson reconstruction
            
        Returns:
            Reconstructed triangle mesh
        """
        # Estimate normals
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        point_cloud.orient_normals_consistent_tangent_plane(k=10)
        
        if method == 'poisson':
            # Poisson surface reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud, depth=depth
            )
            
            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        elif method == 'ball_pivoting':
            # Ball pivoting algorithm
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                point_cloud,
                o3d.utility.DoubleVector(radii)
            )
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
        
        # Clean mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh
    
    def save_reconstruction(self,
                           point_cloud: o3d.geometry.PointCloud,
                           mesh: o3d.geometry.TriangleMesh,
                           output_dir: str = './outputs') -> None:
        """
        Save point cloud and mesh to files.
        
        Args:
            point_cloud: Point cloud to save
            mesh: Mesh to save
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save point cloud
        o3d.io.write_point_cloud(
            os.path.join(output_dir, 'point_cloud.ply'),
            point_cloud
        )
        print(f"Point cloud saved with {len(point_cloud.points)} points")
        
        # Save mesh
        o3d.io.write_triangle_mesh(
            os.path.join(output_dir, 'reconstructed_mesh.ply'),
            mesh
        )
        print(f"Mesh saved with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")


def main():
    """
    Example usage of depth to 3D reconstruction.
    """
    # Load depth map and RGB image
    depth_map = cv2.imread('depth_output.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
    rgb_image = cv2.imread('input_image.jpg')
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # Initialize converter
    converter = DepthTo3DConverter()
    
    # Convert to point cloud
    print("Converting depth map to point cloud...")
    point_cloud = converter.depth_to_point_cloud(depth_map, rgb_image)
    
    # Filter outliers
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Reconstruct mesh
    print("Reconstructing 3D mesh...")
    mesh = converter.reconstruct_mesh(point_cloud, method='poisson', depth=9)
    
    # Save results
    converter.save_reconstruction(point_cloud, mesh)
    
    # Visualize
    print("Visualizing results...")
    o3d.visualization.draw_geometries(
        [point_cloud],
        window_name='Point Cloud',
        width=800,
        height=600
    )
    
    o3d.visualization.draw_geometries(
        [mesh],
        window_name='Reconstructed Mesh',
        width=800,
        height=600
    )


if __name__ == '__main__':
    main()
