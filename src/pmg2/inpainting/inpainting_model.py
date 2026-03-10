import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_inpainting_pipeline(model_id: str = "runwayml/stable-diffusion-inpainting") -> StableDiffusionInpaintPipeline:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe = pipe.to(device)
    return pipe

def run_inpainting(image_path: str, mask_path: str, prompt: str, output_path: str = "inpainted_output.png") -> str:
    pipe = load_inpainting_pipeline()
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    mask = Image.open(mask_path).convert("RGB").resize((512, 512))
    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    result.save(output_path)
    print(f"[INFO] Inpainted image saved to {output_path}")
    return output_path

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        run_inpainting(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python inpainting_model.py <image_path> <mask_path> <prompt>")
