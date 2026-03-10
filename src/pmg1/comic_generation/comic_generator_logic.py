import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to(device)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialogue_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)

def generate_dialogue(prompt: str) -> str:
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt").to(device)
    reply_ids = dialogue_model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

def generate_panel(prompt: str, dialogue: str, panel_index: int) -> Image.Image:
    image = pipe(prompt).images[0]
    return image

def generate_comic(story_prompt: str, num_panels: int = 4, output_path: str = "comic_output.png") -> str:
    panel_prompts = [f"{story_prompt} - scene {i+1}" for i in range(num_panels)]
    panels = []
    for i, prompt in enumerate(panel_prompts):
        dialogue = generate_dialogue(prompt)
        panel = generate_panel(prompt, dialogue, i)
        panels.append(panel)
    # Combine panels horizontally
    w, h = panels[0].size
    comic = Image.new("RGB", (w * num_panels, h))
    for i, panel in enumerate(panels):
        comic.paste(panel, (i * w, 0))
    comic.save(output_path)
    return output_path
