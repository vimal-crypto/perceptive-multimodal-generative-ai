import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import open3d as o3d

class Pix2VoxModel(torch.nn.Module):
    def __init__(self):
        super(Pix2VoxModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

def load_pix2vox_model(model_path):
    model = Pix2VoxModel()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def voxel_to_mesh(voxel):
    voxel = voxel.squeeze().cpu().numpy()
    if len(voxel.shape) == 2:
        voxel = np.expand_dims(voxel, axis=0)
    voxel = (voxel > 0.5).astype(np.float32)
    vertices, faces = [], []
    voxel_size = 1.0 / voxel.shape[0]
    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if voxel[x, y, z] > 0:
                    b = len(vertices)
                    vertices.extend([
                        [x*voxel_size, y*voxel_size, z*voxel_size],
                        [(x+1)*voxel_size, y*voxel_size, z*voxel_size],
                        [x*voxel_size, (y+1)*voxel_size, z*voxel_size],
                        [(x+1)*voxel_size, (y+1)*voxel_size, z*voxel_size],
                        [x*voxel_size, y*voxel_size, (z+1)*voxel_size],
                        [(x+1)*voxel_size, y*voxel_size, (z+1)*voxel_size],
                        [x*voxel_size, (y+1)*voxel_size, (z+1)*voxel_size],
                        [(x+1)*voxel_size, (y+1)*voxel_size, (z+1)*voxel_size],
                    ])
                    faces.extend([
                        [b,b+1,b+2],[b+1,b+3,b+2],[b+4,b+5,b+6],[b+5,b+7,b+6],
                        [b,b+1,b+4],[b+1,b+5,b+4],[b+2,b+3,b+6],[b+3,b+7,b+6],
                        [b,b+2,b+4],[b+2,b+6,b+4],[b+1,b+3,b+5],[b+3,b+7,b+5],
                    ])
    return np.array(vertices), np.array(faces)

def save_voxel_as_obj(voxel, output_path):
    vertices, faces = voxel_to_mesh(voxel)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(output_path, mesh)

def process_images(model_path, input_folder, output_folder):
    model = load_pix2vox_model(model_path)
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, file_name)
            input_tensor = preprocess_image(image_path)
            with torch.no_grad():
                voxel = model(input_tensor)
            obj_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.obj")
            save_voxel_as_obj(voxel, obj_path)
            print(f"[INFO] Saved: {obj_path}")

if __name__ == '__main__':
    process_images("Pix2Vox-A-ShapeNet.pth", "Input", "Output")
