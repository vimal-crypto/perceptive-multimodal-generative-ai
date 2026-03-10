import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from .comic_generator_logic import add_speech_bubble

# Load models lazily to avoid GPU memory issues at import time
_sd_pipeline = None
_dialogue_model = None
_dialogue_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_stable_diffusion():
    global _sd_pipeline
    if _sd_pipeline is None:
        print("[INFO] Loading Stable Diffusion pipeline...")
        _sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32
        ).to(_device)
        _sd_pipeline.safety_checker = None  # disable for research use
    return _sd_pipeline


def _load_dialogue_model():
    global _dialogue_model, _dialogue_tokenizer
    if _dialogue_model is None:
        print("[INFO] Loading DialoGPT dialogue model...")
        model_name = "microsoft/DialoGPT-medium"
        _dialogue_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _dialogue_model = AutoModelForCausalLM.from_pretrained(model_name).to(_device)
    return _dialogue_model, _dialogue_tokenizer


def generate_dialogue_from_prompt(prompt: str) -> str:
    """
    Generate character dialogue from a scene prompt using DialoGPT.

    Args:
        prompt: Scene description to generate contextual dialogue from.

    Returns:
        Generated dialogue string.
    """
    model, tokenizer = _load_dialogue_model()
    input_ids = tokenizer.encode(
        prompt + tokenizer.eos_token, return_tensors="pt"
    ).to(_device)
    reply_ids = model.generate(
        input_ids,
        max_length=80,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.92,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    dialogue = tokenizer.decode(
        reply_ids[:, input_ids.shape[-1]:][0],
        skip_special_tokens=True
    ).strip()
    return dialogue if dialogue else "..."


def generate_scene_with_dialogue(prompt: str, dialogue: str = None, output_path: str = "panel.png") -> str:
    """
    Generate a comic scene image with Stable Diffusion and overlay a dialogue speech bubble.

    Args:
        prompt: Text prompt for image generation.
        dialogue: Dialogue text. If None, auto-generates using DialoGPT.
        output_path: Path to save the final panel image.

    Returns:
        Path to the saved panel image.
    """
    pipeline = _load_stable_diffusion()

    print(f"[INFO] Generating image for prompt: '{prompt}'")
    with torch.autocast(_device) if _device == "cuda" else torch.no_grad():
        image = pipeline(prompt).images[0]

    if dialogue is None:
        print("[INFO] Auto-generating dialogue...")
        dialogue = generate_dialogue_from_prompt(prompt)
    print(f"[INFO] Dialogue: {dialogue}")

    panel = add_speech_bubble(image.convert("RGBA"), dialogue)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    panel.save(output_path)
    print(f"[INFO] Panel saved: {output_path}")
    return output_path
