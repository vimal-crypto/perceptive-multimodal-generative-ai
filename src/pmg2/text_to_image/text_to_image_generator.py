import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

_device = "cuda" if torch.cuda.is_available() else "cpu"


class TextToImageGenerator:
    """
    CLIP-guided latent diffusion text-to-image generator.
    Uses Stable Diffusion v1.4 with CLIP text encoder for prompt-to-image synthesis.

    The generation process follows:
        z = f(CLIP_encode(prompt))  ->  latent representation
        x_0 = Decoder(Diffuse(z, noise))  ->  final image
    """

    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4"):
        print(f"[INFO] Loading Text-to-Image pipeline: {model_id}")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32
        ).to(_device)
        self.pipeline.safety_checker = None

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        output_path: str = None
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Positive text prompt.
            negative_prompt: Negative guidance prompt (things to avoid).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            width, height: Output image dimensions.
            output_path: Optional path to save the image.

        Returns:
            Generated PIL Image.
        """
        print(f"[INFO] Generating image for: '{prompt}'")
        context = torch.autocast(_device) if _device == "cuda" else torch.no_grad()
        with context:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            result.save(output_path)
            print(f"[INFO] Image saved: {output_path}")
        return result

    def generate_batch(
        self,
        prompts: list,
        output_dir: str = "outputs/text_to_image"
    ) -> list:
        """
        Generate images for a list of prompts.

        Args:
            prompts: List of text prompts.
            output_dir: Directory to save generated images.

        Returns:
            List of paths to saved images.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, prompt in enumerate(prompts):
            output_path = os.path.join(output_dir, f"generated_{i+1}.png")
            self.generate(prompt, output_path=output_path)
            paths.append(output_path)
        return paths


def generate_image_from_text(
    prompt: str,
    output_path: str = "outputs/generated.png",
    num_inference_steps: int = 50
) -> str:
    """
    Convenience function: generate and save a single image from a text prompt.

    Args:
        prompt: Text description of the image.
        output_path: Path to save the generated image.
        num_inference_steps: Denoising steps.

    Returns:
        Path to the saved image.
    """
    gen = TextToImageGenerator()
    gen.generate(prompt, output_path=output_path, num_inference_steps=num_inference_steps)
    return output_path
