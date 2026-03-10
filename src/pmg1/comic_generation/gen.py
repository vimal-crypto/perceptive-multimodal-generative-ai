import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    return pipe(prompt).images[0]

def generate_dialogue(prompt):
    device = next(model.parameters()).device
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt").to(device)
    reply_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

def add_dialogue_to_image(image, dialogue, bubble_coords):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bubble_coords, outline="black", width=3, fill="white")
    font = ImageFont.load_default()
    text_position = (bubble_coords[0][0] + 10, bubble_coords[0][1] + 10)
    draw.text(text_position, dialogue, font=font, fill="black")
    return image

def create_comic_sequence(prompts, dialogues, bubble_coords_list, output_path="comic_sequence.png"):
    panels = []
    for i in range(len(prompts)):
        image = generate_image(prompts[i])
        dialogue = generate_dialogue(dialogues[i])
        panel = add_dialogue_to_image(image, dialogue, bubble_coords_list[i])
        panels.append(panel)
    w, h = panels[0].size
    collage = Image.new("RGB", (w * len(panels), h))
    for i, panel in enumerate(panels):
        collage.paste(panel, (i * w, 0))
    collage.save(output_path)
    return collage
