import os
import numpy as np
import open3d as o3d


def depth_to_point_cloud(
    depth_map: np.ndarray,
    rgb_image: np.ndarray = None,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = None,
    cy: float = None,
    depth_scale: float = 1.0
) -> o3d.geometry.PointCloud:
    """
    Convert a depth map to a 3D point cloud using pinhole camera projection.

    The projection formula for each pixel (u, v) with depth d is:
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

    Args:
        depth_map: HxW float32 depth array (normalized or metric).
        rgb_image: Optional HxWx3 uint8 RGB array for colorized point cloud.
        fx, fy: Focal lengths in pixels (default 500).
        cx, cy: Principal point. Defaults to image center.
        depth_scale: Scale factor applied to raw depth values.

    Returns:
        Open3D PointCloud object.
    """
    h, w = depth_map.shape
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            d = depth_map[v, u] * depth_scale
            if d <= 0:
                continue
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d
            points.append([X, Y, Z])
            if rgb_image is not None:
                colors.append(rgb_image[v, u] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd


def save_point_cloud_as_ply(pcd: o3d.geometry.PointCloud, output_path: str) -> str:
    """
    Save an Open3D PointCloud to a .ply file.

    Args:
        pcd: Open3D PointCloud object.
        output_path: Path to save the .ply file.

    Returns:
        Path to saved file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[INFO] Point cloud saved: {output_path}")
    return output_path


def remove_outliers(pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0):
    """
    Remove statistical outliers from a point cloud.

    Args:
        pcd: Input point cloud.
        nb_neighbors: Number of nearest neighbors for outlier detection.
        std_ratio: Standard deviation multiplier threshold.

    Returns:
        Cleaned PointCloud.
    """
    cleaned, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cleaned


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.1, max_nn: int = 30):
    """
    Estimate surface normals for a point cloud.

    Args:
        pcd: Input point cloud.
        radius: Neighborhood radius for normal estimation.
        max_nn: Maximum number of nearest neighbors.

    Returns:
        PointCloud with normals computed.
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return pcd
