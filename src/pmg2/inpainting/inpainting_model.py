import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

_device = "cuda" if torch.cuda.is_available() else "cpu"
_inpaint_pipeline = None


def _load_inpaint_pipeline():
    global _inpaint_pipeline
    if _inpaint_pipeline is None:
        print("[INFO] Loading Stable Diffusion Inpainting pipeline...")
        _inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32
        ).to(_device)
        _inpaint_pipeline.safety_checker = None
    return _inpaint_pipeline


class InpaintingModel:
    """
    Wrapper for Stable Diffusion inpainting.
    Fills masked regions of an image guided by a text prompt.
    """

    def __init__(self):
        self.pipeline = _load_inpaint_pipeline()

    def inpaint(
        self,
        image_path: str,
        mask_path: str,
        prompt: str,
        output_path: str = "outputs/inpainted.png",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> str:
        """
        Inpaint the masked region of an image using a text prompt.

        Args:
            image_path: Path to the original image.
            mask_path: Path to the binary mask image (white = region to inpaint).
            prompt: Text description of what to fill in.
            output_path: Path to save the inpainted result.
            num_inference_steps: Diffusion denoising steps.
            guidance_scale: Classifier-free guidance scale.

        Returns:
            Path to the saved inpainted image.
        """
        image = Image.open(image_path).convert("RGB").resize((512, 512))
        mask = Image.open(mask_path).convert("RGB").resize((512, 512))

        print(f"[INFO] Inpainting with prompt: '{prompt}'")
        with torch.autocast(_device) if _device == "cuda" else torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        result.save(output_path)
        print(f"[INFO] Inpainted image saved: {output_path}")
        return output_path


def inpaint_image(
    image_path: str,
    mask_path: str,
    prompt: str,
    output_path: str = "outputs/inpainted.png"
) -> str:
    """
    Convenience function to inpaint an image without instantiating the class.

    Args:
        image_path: Path to original image.
        mask_path: Path to binary mask.
        prompt: Text prompt for inpainting.
        output_path: Output path.

    Returns:
        Path to saved inpainted image.
    """
    model = InpaintingModel()
    return model.inpaint(image_path, mask_path, prompt, output_path)
