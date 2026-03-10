import os
import torch
import numpy as np
import cv2
from PIL import Image
from .pix2vox_model import load_pix2vox_model, preprocess_image, save_voxel_as_obj
from .depth_estimator import MiDaSDepthEstimator
from .point_cloud_utils import depth_to_point_cloud, save_point_cloud_as_ply, remove_outliers, estimate_normals


def run_reconstruction_pipeline(
    image_path: str,
    output_dir: str = "outputs/3d",
    pix2vox_model_path: str = None,
    use_depth: bool = True,
    use_voxel: bool = True,
    device: str = None
) -> dict:
    """
    Full 2D-to-3D reconstruction pipeline.
    Given an input image, produces:
        - MiDaS depth map (.png)
        - Open3D point cloud from depth (.ply)
        - Pix2Vox voxel mesh (.obj) [if pix2vox_model_path provided]

    Args:
        image_path: Path to input RGB image.
        output_dir: Directory to save outputs.
        pix2vox_model_path: Optional path to Pix2Vox checkpoint.
        use_depth: Whether to run MiDaS depth estimation.
        use_voxel: Whether to run Pix2Vox voxel prediction.
        device: Compute device.

    Returns:
        Dict with keys: 'depth_path', 'point_cloud_path', 'obj_path'.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    results = {}

    # Step 1: MiDaS Depth Estimation
    if use_depth:
        print("[PIPELINE] Step 1: Depth Estimation...")
        depth_estimator = MiDaSDepthEstimator(device=device)
        depth_map = depth_estimator.estimate(image_path)
        depth_path = os.path.join(output_dir, f"{base_name}_depth.png")
        depth_estimator.save_depth_map(depth_map, depth_path)
        results['depth_path'] = depth_path

        # Step 2: Depth to Point Cloud
        print("[PIPELINE] Step 2: Generating Point Cloud...")
        rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        pcd = depth_to_point_cloud(depth_map, rgb_image=rgb)
        pcd = remove_outliers(pcd)
        pcd = estimate_normals(pcd)
        ply_path = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
        save_point_cloud_as_ply(pcd, ply_path)
        results['point_cloud_path'] = ply_path

    # Step 3: Pix2Vox Voxel Prediction
    if use_voxel and pix2vox_model_path and os.path.exists(pix2vox_model_path):
        print("[PIPELINE] Step 3: Voxel Reconstruction...")
        model = load_pix2vox_model(pix2vox_model_path, device)
        tensor = preprocess_image(image_path).to(device)
        with torch.no_grad():
            voxel = model(tensor)
        obj_path = os.path.join(output_dir, f"{base_name}_mesh.obj")
        save_voxel_as_obj(voxel, obj_path)
        results['obj_path'] = obj_path

    print(f"[PIPELINE] Complete. Outputs: {results}")
    return results
