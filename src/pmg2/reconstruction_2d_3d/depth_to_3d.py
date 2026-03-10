import numpy as np
import open3d as o3d

def depth_to_3d(depth_map: np.ndarray, fx: float = 500.0, fy: float = 500.0) -> o3d.geometry.PointCloud:
    h, w = depth_map.shape
    cx, cy = w / 2.0, h / 2.0
    points = []
    for v in range(h):
        for u in range(w):
            z = depth_map[v, u]
            if z > 0:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    return pcd

if __name__ == '__main__':
    import cv2
    depth = cv2.imread('depth_output.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    pcd = depth_to_3d(depth)
    o3d.io.write_point_cloud('output_from_depth.ply', pcd)
    print('Point cloud saved.')
