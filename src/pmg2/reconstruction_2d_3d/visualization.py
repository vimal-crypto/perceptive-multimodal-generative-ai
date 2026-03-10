import os
import numpy as np
import open3d as o3d
from PIL import Image


def visualize_point_cloud(ply_path: str, window_title: str = "Point Cloud"):
    """
    Open an interactive Open3D viewer for a .ply point cloud.

    Args:
        ply_path: Path to the .ply file.
        window_title: Window title for the viewer.
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"[INFO] Visualizing: {ply_path} ({len(pcd.points)} points)")
    o3d.visualization.draw_geometries([pcd], window_name=window_title)


def visualize_mesh(obj_path: str, window_title: str = "3D Mesh"):
    """
    Open an interactive Open3D viewer for a .obj mesh file.

    Args:
        obj_path: Path to the .obj file.
        window_title: Window title for the viewer.
    """
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    print(f"[INFO] Visualizing mesh: {obj_path}")
    o3d.visualization.draw_geometries([mesh], window_name=window_title)


def render_depth_comparison(original_path: str, depth_path: str, output_path: str = None) -> Image.Image:
    """
    Create a side-by-side comparison of original image and depth map.

    Args:
        original_path: Path to original RGB image.
        depth_path: Path to depth map image.
        output_path: Optional path to save the comparison.

    Returns:
        Side-by-side PIL Image.
    """
    orig = Image.open(original_path).convert("RGB")
    depth = Image.open(depth_path).convert("RGB")

    target_h = min(orig.height, depth.height, 400)
    orig = orig.resize((int(orig.width * target_h / orig.height), target_h))
    depth = depth.resize((int(depth.width * target_h / depth.height), target_h))

    combined = Image.new("RGB", (orig.width + depth.width + 10, target_h), (30, 30, 30))
    combined.paste(orig, (0, 0))
    combined.paste(depth, (orig.width + 10, 0))

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        combined.save(output_path)
        print(f"[INFO] Comparison saved: {output_path}")
    return combined


def visualize_3d_output(results: dict):
    """
    Visualize all 3D outputs produced by the reconstruction pipeline.

    Args:
        results: Dict from run_reconstruction_pipeline with keys
                 'depth_path', 'point_cloud_path', 'obj_path'.
    """
    if 'point_cloud_path' in results:
        visualize_point_cloud(results['point_cloud_path'])
    if 'obj_path' in results:
        visualize_mesh(results['obj_path'])
