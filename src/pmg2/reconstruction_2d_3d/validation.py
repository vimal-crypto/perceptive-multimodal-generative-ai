import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(input_image_path, depth_map_path):
    input_image = cv2.imread(input_image_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    depth_map_normalized = cv2.normalize(depth_map, None, alpha=0, beta=1,
                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return input_image, depth_map_normalized

def display_images(input_image, depth_map_normalized):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(depth_map_normalized, cmap='plasma')
    plt.title('Normalized Depth Map')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    input_image, depth_map_normalized = load_images('input.jpg', 'depth_output.jpg')
    display_images(input_image, depth_map_normalized)
