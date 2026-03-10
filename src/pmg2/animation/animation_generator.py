import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline(model_id: str = "CompVis/stable-diffusion-v1-4") -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    return pipe.to(device)

def interpolate_prompts(prompt_a: str, prompt_b: str, steps: int = 5) -> list:
    """Simple linear interpolation between two text prompts for frame generation."""
    return [
        f"{prompt_a}" if i == 0 else
        f"{prompt_b}" if i == steps - 1 else
        f"{prompt_a}, transitioning to {prompt_b}, step {i} of {steps}"
        for i in range(steps)
    ]

def generate_animation_frames(prompt_a: str, prompt_b: str, output_dir: str = "frames",
                               num_frames: int = 5) -> list:
    os.makedirs(output_dir, exist_ok=True)
    pipe = load_pipeline()
    prompts = interpolate_prompts(prompt_a, prompt_b, num_frames)
    frame_paths = []
    for i, prompt in enumerate(prompts):
        image = pipe(prompt).images[0]
        path = os.path.join(output_dir, f"frame_{i:03d}.png")
        image.save(path)
        frame_paths.append(path)
        print(f"[INFO] Frame {i+1}/{num_frames} saved: {path}")
    return frame_paths

def frames_to_gif(frame_paths: list, output_path: str = "animation.gif", duration: int = 200):
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   optimize=False, duration=duration, loop=0)
    print(f"[INFO] Animation saved to {output_path}")

if __name__ == '__main__':
    paths = generate_animation_frames(
        prompt_a="A peaceful forest at dawn",
        prompt_b="A forest engulfed in flames at night",
        num_frames=6
    )
    frames_to_gif(paths)
