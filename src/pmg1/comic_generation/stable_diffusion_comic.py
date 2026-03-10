import os
import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont

_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(sd_unet_path=None, sd_vae_path=None, sd_text_encoder_path=None):
    """
    Load Stable Diffusion and GPT-2 dialogue models.
    Optionally load custom fine-tuned weights for UNet, VAE, and text encoder.

    Returns:
        Tuple of (sd_pipeline, dialogue_model, dialogue_tokenizer)
    """
    print("[INFO] Loading Stable Diffusion...")
    sd_model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32
    )
    if sd_unet_path and os.path.exists(sd_unet_path):
        sd_model.unet.load_state_dict(torch.load(sd_unet_path, map_location=_device))
    if sd_vae_path and os.path.exists(sd_vae_path):
        sd_model.vae.load_state_dict(torch.load(sd_vae_path, map_location=_device))
    if sd_text_encoder_path and os.path.exists(sd_text_encoder_path):
        sd_model.text_encoder.load_state_dict(torch.load(sd_text_encoder_path, map_location=_device))
    sd_model = sd_model.to(_device)
    sd_model.safety_checker = None

    print("[INFO] Loading GPT-2 dialogue model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dialogue_model = AutoModelForCausalLM.from_pretrained("gpt2").to(_device)

    return sd_model, dialogue_model, tokenizer


def generate_dialogue(prompt: str, model, tokenizer) -> str:
    """Generate dialogue using GPT-2."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(_device)
    output = model.generate(
        input_ids,
        max_length=60,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.92,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def add_textbox_to_image(image: Image.Image, text: str) -> Image.Image:
    """Overlay a wrapped text box on the top-left of the image."""
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    img_w, img_h = image.size
    padding = 16
    box_max_w = int(img_w * 0.5)

    words = text.split()
    lines, current = [], []
    for word in words:
        current.append(word)
        bbox = draw.textbbox((0, 0), " ".join(current), font=font)
        if bbox[2] > box_max_w - 2 * padding:
            if len(current) > 1:
                lines.append(" ".join(current[:-1]))
                current = [word]
    if current:
        lines.append(" ".join(current))

    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + 4
    box_h = line_h * len(lines) + 2 * padding

    draw.rounded_rectangle(
        [(padding, padding), (padding + box_max_w, padding + box_h)],
        radius=8, fill=(255, 255, 255, 200), outline=(0, 0, 0, 255), width=2
    )
    y = padding * 2
    for line in lines:
        draw.text((padding * 2, y), line, font=font, fill=(0, 0, 0, 255))
        y += line_h

    return Image.alpha_composite(image, overlay)


def generate_scene_with_generated_dialogue(
    scene_prompt: str,
    output_path: str,
    sd_model=None,
    dialogue_model=None,
    dialogue_tokenizer=None
) -> str:
    """
    Generate a comic panel: create a scene with Stable Diffusion,
    auto-generate dialogue with GPT-2, and overlay it as a text box.

    Args:
        scene_prompt: Description of the comic scene.
        output_path: Where to save the output panel.
        sd_model: Pre-loaded SD pipeline (loaded on first call if None).
        dialogue_model: Pre-loaded GPT-2 model (loaded on first call if None).
        dialogue_tokenizer: Pre-loaded GPT-2 tokenizer (loaded on first call if None).

    Returns:
        Path to the saved panel image.
    """
    if sd_model is None or dialogue_model is None:
        sd_model, dialogue_model, dialogue_tokenizer = load_models()

    print(f"[INFO] Generating scene: {scene_prompt}")
    scene_image = sd_model(scene_prompt).images[0]

    dialogue = generate_dialogue(scene_prompt, dialogue_model, dialogue_tokenizer)
    print(f"[INFO] Generated dialogue: {dialogue}")

    panel = add_textbox_to_image(scene_image, dialogue)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    panel.save(output_path)
    print(f"[INFO] Panel saved: {output_path}")
    return output_path
