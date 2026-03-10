import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

device = "cuda" if torch.cuda.is_available() else "cpu"
model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
model = model.to(device)

def find_textbox_position(image, padding=10):
    img_width, img_height = image.size
    box_width = int(img_width * 0.4)
    box_height = int(img_height * 0.2)
    position = (img_width - box_width - padding, padding)
    return position, box_width, box_height

def add_textbox_to_image(image, text):
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    position, box_width, box_height = find_textbox_position(image)
    txt_image = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_image)
    box_position = [position[0], position[1], position[0] + box_width, position[1] + box_height]
    draw.rectangle(box_position, fill=(255, 255, 255, 180), outline="black")
    draw.text((position[0] + 10, position[1] + 10), text, font=font, fill="black")
    return Image.alpha_composite(image, txt_image)

def generate_scene_with_dialogue(prompt, dialogue, output_path):
    scene_image = model(prompt).images[0].convert("RGBA")
    scene_with_text = add_textbox_to_image(scene_image, dialogue)
    scene_with_text.save(output_path)
    print(f"Scene saved as {output_path}")

def create_comic_strip(scene_paths, output_path="comic_strip.png"):
    images = [Image.open(scene) for scene in scene_paths]
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    comic_strip = Image.new('RGBA', (total_width, max_height), (255, 255, 255, 0))
    x_offset = 0
    for img in images:
        comic_strip.paste(img, (x_offset, 0))
        x_offset += img.width
    comic_strip.save(output_path)
    print(f"Comic strip saved as {output_path}")
