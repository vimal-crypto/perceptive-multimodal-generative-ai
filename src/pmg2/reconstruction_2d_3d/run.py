import argparse
from pix2vox import process_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run 2D to 3D reconstruction')
    parser.add_argument('--model', type=str, default='Pix2Vox-A-ShapeNet.pth', help='Path to model checkpoint')
    parser.add_argument('--input', type=str, default='Input', help='Input folder with images')
    parser.add_argument('--output', type=str, default='Output', help='Output folder for .obj files')
    args = parser.parse_args()
    process_images(args.model, args.input, args.output)
