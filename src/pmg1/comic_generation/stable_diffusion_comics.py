"""
Stable Diffusion Comic Generation Module for PMG-1

This module provides functionality for generating comic-style images using
Stable Diffusion with persona-based prompts and GPT integration.

Author: PMG-AI Team
Date: 2024
"""

import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
from typing import List, Optional, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StableDiffusionComicGenerator:
    """
    Comic generation system using Stable Diffusion with persona-driven prompts.
    
    This class implements comic panel generation with consistent character styling,
    narrative coherence, and GPT-integrated persona systems.
    
    Attributes:
        model_id (str): Hugging Face model identifier for Stable Diffusion
        device (str): Compute device ('cuda' or 'cpu')
        pipeline: Loaded Stable Diffusion pipeline
        guidance_scale (float): Classifier-free guidance scale
        num_inference_steps (int): Number of denoising steps
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ):
        """
        Initialize the Stable Diffusion comic generator.
        
        Args:
            model_id: Hugging Face model identifier
            device: Compute device ('cuda' or 'cpu')
            guidance_scale: Classifier-free guidance scale (default: 7.5)
            num_inference_steps: Number of denoising steps (default: 50)
        """
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        
        logger.info(f"Initializing Stable Diffusion on {self.device}")
        self._load_pipeline()
    
    def _load_pipeline(self) -> None:
        """
        Load the Stable Diffusion pipeline with optimized settings.
        """
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            )
            
            # Use DPM-Solver++ for faster generation
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory optimizations
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
            
            logger.info("Stable Diffusion pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def generate_comic_panel(
        self,
        prompt: str,
        negative_prompt: str = "ugly, blurry, low quality, distorted",
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        comic_style: str = "anime"
    ) -> Image.Image:
        """
        Generate a single comic panel from a text prompt.
        
        Args:
            prompt: Text description of the comic panel
            negative_prompt: Elements to avoid in generation
            width: Output image width in pixels
            height: Output image height in pixels
            seed: Random seed for reproducibility
            comic_style: Style of comic (anime, manga, western, etc.)
        
        Returns:
            PIL Image object containing the generated panel
        """
        # Enhance prompt with comic styling
        style_prefix = self._get_style_prefix(comic_style)
        enhanced_prompt = f"{style_prefix}, {prompt}, comic panel, detailed artwork"
        
        logger.info(f"Generating comic panel: {enhanced_prompt[:100]}...")
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate image
        with torch.autocast(self.device):
            result = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator
            )
        
        return result.images[0]
    
    def generate_comic_sequence(
        self,
        prompts: List[str],
        persona_context: Optional[Dict] = None,
        panel_size: Tuple[int, int] = (512, 512),
        seed: Optional[int] = None,
        comic_style: str = "anime"
    ) -> List[Image.Image]:
        """
        Generate a sequence of comic panels with narrative consistency.
        
        Args:
            prompts: List of text descriptions for each panel
            persona_context: Character and setting context for consistency
            panel_size: (width, height) tuple for each panel
            seed: Base random seed for sequence
            comic_style: Style of comic
        
        Returns:
            List of PIL Image objects for each panel
        """
        panels = []
        base_seed = seed if seed is not None else np.random.randint(0, 1000000)
        
        # Apply persona context to prompts if provided
        if persona_context:
            prompts = self._apply_persona_context(prompts, persona_context)
        
        logger.info(f"Generating comic sequence with {len(prompts)} panels")
        
        for i, prompt in enumerate(prompts):
            panel_seed = base_seed + i
            panel = self.generate_comic_panel(
                prompt=prompt,
                width=panel_size[0],
                height=panel_size[1],
                seed=panel_seed,
                comic_style=comic_style
            )
            panels.append(panel)
            logger.info(f"Generated panel {i+1}/{len(prompts)}")
        
        return panels
    
    def _get_style_prefix(self, style: str) -> str:
        """
        Get style-specific prompt prefix.
        
        Args:
            style: Comic style identifier
        
        Returns:
            Prompt prefix for the specified style
        """
        style_prefixes = {
            "anime": "anime style, vibrant colors",
            "manga": "manga style, black and white, screentone",
            "western": "western comic book style, bold lines",
            "watercolor": "watercolor comic style, soft edges",
            "realistic": "realistic comic style, detailed rendering"
        }
        return style_prefixes.get(style, "comic book style")
    
    def _apply_persona_context(
        self,
        prompts: List[str],
        context: Dict
    ) -> List[str]:
        """
        Apply character persona and setting context to prompts.
        
        Args:
            prompts: List of base prompts
            context: Dictionary containing character and setting info
        
        Returns:
            Enhanced prompts with persona context
        """
        character_desc = context.get("character", "")
        setting_desc = context.get("setting", "")
        
        enhanced_prompts = []
        for prompt in prompts:
            enhanced = prompt
            if character_desc:
                enhanced = f"{character_desc}, {enhanced}"
            if setting_desc:
                enhanced = f"{enhanced}, {setting_desc}"
            enhanced_prompts.append(enhanced)
        
        return enhanced_prompts
    
    def save_panel(self, panel: Image.Image, output_path: str) -> None:
        """
        Save a comic panel to disk.
        
        Args:
            panel: PIL Image object
            output_path: Path to save the image
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        panel.save(output_path)
        logger.info(f"Panel saved to {output_path}")
    
    def create_comic_strip(
        self,
        panels: List[Image.Image],
        layout: str = "horizontal",
        spacing: int = 10,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Combine multiple panels into a comic strip.
        
        Args:
            panels: List of panel images
            layout: 'horizontal' or 'vertical' layout
            spacing: Pixels between panels
            background_color: RGB tuple for background
        
        Returns:
            Combined comic strip image
        """
        if not panels:
            raise ValueError("No panels provided")
        
        panel_width, panel_height = panels[0].size
        
        if layout == "horizontal":
            strip_width = (panel_width * len(panels)) + (spacing * (len(panels) - 1))
            strip_height = panel_height
        else:  # vertical
            strip_width = panel_width
            strip_height = (panel_height * len(panels)) + (spacing * (len(panels) - 1))
        
        strip = Image.new('RGB', (strip_width, strip_height), background_color)
        
        x_offset = 0
        y_offset = 0
        
        for panel in panels:
            strip.paste(panel, (x_offset, y_offset))
            if layout == "horizontal":
                x_offset += panel_width + spacing
            else:
                y_offset += panel_height + spacing
        
        logger.info(f"Created {layout} comic strip with {len(panels)} panels")
        return strip


def main():
    """
    Example usage of the Stable Diffusion comic generator.
    """
    # Initialize generator
    generator = StableDiffusionComicGenerator(device="cuda")
    
    # Define character persona
    persona = {
        "character": "a young wizard with blue robes and a staff",
        "setting": "in a magical forest with glowing mushrooms"
    }
    
    # Define story prompts
    prompts = [
        "character discovers a mysterious portal",
        "character casts a spell to open the portal",
        "character steps through into unknown realm"
    ]
    
    # Generate comic sequence
    panels = generator.generate_comic_sequence(
        prompts=prompts,
        persona_context=persona,
        comic_style="anime",
        seed=42
    )
    
    # Save individual panels
    for i, panel in enumerate(panels):
        generator.save_panel(panel, f"outputs/comic/panel_{i+1}.png")
    
    # Create comic strip
    strip = generator.create_comic_strip(panels, layout="horizontal")
    generator.save_panel(strip, "outputs/comic/comic_strip.png")
    
    print("Comic generation complete!")


if __name__ == "__main__":
    main()
