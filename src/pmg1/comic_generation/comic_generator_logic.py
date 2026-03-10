import os
from PIL import Image, ImageDraw, ImageFont
from .scene_generator import generate_scene_with_dialogue


def generate_comic_panel(prompt: str, dialogue: str, output_path: str) -> str:
    """
    Generate a single comic panel with image and dialogue text box.

    Args:
        prompt: Scene description for Stable Diffusion image generation.
        dialogue: Character dialogue. If empty, auto-generated from the prompt.
        output_path: File path to save the generated panel PNG.

    Returns:
        Path to the saved panel image.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    generate_scene_with_dialogue(prompt, dialogue if dialogue else None, output_path)
    return output_path


def create_comic_strip(panel_paths: list, output_path: str = "outputs/comic_strip.png") -> str:
    """
    Combine multiple comic panels into a single horizontal comic strip.

    Args:
        panel_paths: List of file paths to individual panel images.
        output_path: File path to save the combined comic strip.

    Returns:
        Path to the saved comic strip.
    """
    if not panel_paths:
        raise ValueError("panel_paths cannot be empty.")

    images = [Image.open(p).convert("RGBA") for p in panel_paths]
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    comic_strip = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 255))
    x_offset = 0
    for img in images:
        comic_strip.paste(img, (x_offset, 0))
        x_offset += img.width

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    comic_strip.save(output_path)
    print(f"[INFO] Comic strip saved: {output_path}")
    return output_path


def add_speech_bubble(image: Image.Image, text: str, position: tuple = None) -> Image.Image:
    """
    Add a speech bubble with wrapped text to a PIL image.

    Args:
        image: PIL Image object.
        text: Text to display inside the bubble.
        position: (x, y) top-left of the bubble. Defaults to top-left corner.

    Returns:
        Modified PIL Image with speech bubble.
    """
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    img_w, img_h = image.size
    padding = 12
    box_max_w = int(img_w * 0.55)
    pos = position if position else (padding, padding)

    # Word-wrap
    words = text.split()
    lines, current = [], []
    for word in words:
        current.append(word)
        test_line = " ".join(current)
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] > box_max_w - 2 * padding:
            if len(current) > 1:
                lines.append(" ".join(current[:-1]))
                current = [word]
            else:
                lines.append(test_line)
                current = []
    if current:
        lines.append(" ".join(current))

    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 4
    box_h = line_height * len(lines) + 2 * padding
    box_w = box_max_w

    draw.rounded_rectangle(
        [pos, (pos[0] + box_w, pos[1] + box_h)],
        radius=10,
        fill=(255, 255, 255, 210),
        outline=(0, 0, 0, 255),
        width=2
    )

    text_y = pos[1] + padding
    for line in lines:
        draw.text((pos[0] + padding, text_y), line, font=font, fill=(0, 0, 0, 255))
        text_y += line_height

    return Image.alpha_composite(image, overlay)
