import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import open3d as o3d


class Pix2VoxModel(nn.Module):
    """
    Simplified Pix2Vox encoder-decoder architecture for 2D-to-3D voxel prediction.
    Encodes an RGB image into feature maps, then decodes to a voxel occupancy grid.

    Architecture:
        Encoder: Conv2d(3->64) -> Conv2d(64->128)
        Decoder: ConvTranspose2d(128->64) -> ConvTranspose2d(64->32) -> ConvTranspose2d(32->1) + Sigmoid
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode image to feature map, decode to voxel occupancy.

        Args:
            x: Batch of RGB images, shape (B, 3, H, W).

        Returns:
            Voxel occupancy map, shape (B, 1, H, W), values in [0, 1].
        """
        features = self.encoder(x)
        output = self.decoder(features)
        return output


def load_pix2vox_model(model_path: str, device: str = None) -> Pix2VoxModel:
    """
    Load a Pix2Vox model from a .pth checkpoint.
    Handles DataParallel 'module.' prefix stripping automatically.

    Args:
        model_path: Path to the checkpoint file.
        device: Device to load to. Auto-detects if None.

    Returns:
        Loaded Pix2VoxModel in eval mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Pix2VoxModel()
    checkpoint = torch.load(model_path, map_location=device)

    # Strip DataParallel 'module.' prefix
    new_state = {k[7:] if k.startswith("module.") else k: v for k, v in checkpoint.items()}
    model.load_state_dict(new_state, strict=False)
    model.eval()
    model.to(device)
    print(f"[INFO] Pix2Vox model loaded from: {model_path}")
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess an image for Pix2Vox inference.
    Resizes to 224x224, converts to tensor, and normalizes to [-1, 1].

    Args:
        image_path: Path to the input image.

    Returns:
        Preprocessed image tensor of shape (1, 3, 224, 224).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def voxel_to_mesh(voxel: torch.Tensor, threshold: float = 0.5) -> tuple:
    """
    Convert a voxel occupancy tensor to mesh vertices and faces.
    Handles 2D output by adding a dummy depth dimension.

    Args:
        voxel: Voxel tensor, any shape.
        threshold: Occupancy threshold (cells above this are considered filled).

    Returns:
        Tuple of (vertices: np.ndarray, faces: np.ndarray).
    """
    voxel_np = voxel.squeeze().cpu().detach().numpy()
    if voxel_np.ndim == 2:
        voxel_np = np.expand_dims(voxel_np, axis=0)
    if voxel_np.ndim != 3:
        raise ValueError(f"Expected 3D voxel, got shape {voxel_np.shape}")

    occupied = (voxel_np > threshold).astype(np.float32)
    voxel_size = 1.0 / max(occupied.shape)
    vertices, faces = [], []

    for x in range(occupied.shape[0]):
        for y in range(occupied.shape[1]):
            for z in range(occupied.shape[2]):
                if occupied[x, y, z] > 0:
                    vi = len(vertices)
                    vx, vy, vz = x * voxel_size, y * voxel_size, z * voxel_size
                    vs = voxel_size
                    vertices.extend([
                        [vx, vy, vz], [vx+vs, vy, vz], [vx, vy+vs, vz], [vx+vs, vy+vs, vz],
                        [vx, vy, vz+vs], [vx+vs, vy, vz+vs], [vx, vy+vs, vz+vs], [vx+vs, vy+vs, vz+vs]
                    ])
                    faces.extend([
                        [vi, vi+1, vi+2], [vi+1, vi+3, vi+2],
                        [vi+4, vi+5, vi+6], [vi+5, vi+7, vi+6],
                        [vi, vi+1, vi+4], [vi+1, vi+5, vi+4],
                        [vi+2, vi+3, vi+6], [vi+3, vi+7, vi+6],
                        [vi, vi+2, vi+4], [vi+2, vi+6, vi+4],
                        [vi+1, vi+3, vi+5], [vi+3, vi+7, vi+5],
                    ])
    return np.array(vertices), np.array(faces)


def save_voxel_as_obj(voxel: torch.Tensor, output_path: str) -> str:
    """
    Save a voxel tensor as a .obj 3D mesh file using Open3D.

    Args:
        voxel: Voxel tensor from model output.
        output_path: Output .obj file path.

    Returns:
        Path to the saved .obj file.
    """
    vertices, faces = voxel_to_mesh(voxel)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"[INFO] OBJ saved: {output_path}")
    return output_path


def process_images(
    model_path: str,
    input_folder: str,
    output_folder: str,
    device: str = None
) -> list:
    """
    Batch-process all images in a folder through Pix2Vox to produce .obj files.

    Args:
        model_path: Path to .pth checkpoint.
        input_folder: Folder containing input images.
        output_folder: Folder to save .obj outputs.
        device: Compute device (auto-detected if None).

    Returns:
        List of output .obj file paths.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_pix2vox_model(model_path, device)
    os.makedirs(output_folder, exist_ok=True)
    output_paths = []

    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported):
            img_path = os.path.join(input_folder, filename)
            print(f"[INFO] Processing: {filename}")
            input_tensor = preprocess_image(img_path).to(device)
            with torch.no_grad():
                voxel = model(input_tensor)
            obj_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.obj")
            save_voxel_as_obj(voxel, obj_path)
            output_paths.append(obj_path)

    print(f"[INFO] Processed {len(output_paths)} images.")
    return output_paths
