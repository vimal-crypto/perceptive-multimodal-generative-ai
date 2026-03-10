import cv2
import numpy as np
import open3d as o3d
import torch
from torchvision import transforms
from PIL import Image

def load_depth_model():
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    model.eval()
    return model

def estimate_depth(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        depth = model(input_tensor)
    depth = depth.squeeze().numpy()
    depth = cv2.resize(depth, (image.width, image.height))
    return depth, np.array(image)

def depth_to_pointcloud(depth, image_rgb, fx=500, fy=500):
    h, w = depth.shape
    cx, cy = w / 2, h / 2
    points, colors = [], []
    for v in range(h):
        for u in range(w):
            z = depth[v, u]
            if z > 0:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                colors.append(image_rgb[v, u] / 255.0)
    return np.array(points), np.array(colors)

def save_pointcloud(points, colors, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[INFO] Point cloud saved: {output_path}")

if __name__ == '__main__':
    model = load_depth_model()
    depth, image_rgb = estimate_depth(model, 'input.jpg')
    points, colors = depth_to_pointcloud(depth, image_rgb)
    save_pointcloud(points, colors, 'output.ply')
