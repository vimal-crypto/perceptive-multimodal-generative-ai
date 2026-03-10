import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

def estimate_depth_midas(image_path: str, output_path: str = 'depth_output.jpg') -> np.ndarray:
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        depth = model(input_tensor)
    depth_np = depth.squeeze().numpy()
    depth_np = cv2.resize(depth_np, (image.width, image.height))
    depth_normalized = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(output_path, depth_normalized)
    print(f'Depth map saved to {output_path}')
    return depth_np

if __name__ == '__main__':
    estimate_depth_midas('input.jpg')
