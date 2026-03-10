import os
from PIL import Image, ImageDraw, ImageFont
from .scene_generator import generate_scene_with_dialogue


def create_comic_sequence(
    image_prompts: list,
    dialogue_prompts: list,
    bubble_coords_list: list = None,
    output_dir: str = "outputs",
    final_output: str = "outputs/comic_sequence.png"
) -> str:
    """
    Generate a full comic sequence: one panel per (prompt, dialogue) pair,
    then combine them all into a single strip.

    Args:
        image_prompts: List of scene description strings for each panel.
        dialogue_prompts: List of dialogue strings for each panel.
        bubble_coords_list: Optional list of (x, y) tuples for bubble placement.
        output_dir: Directory to store individual panels.
        final_output: Path for the combined comic strip.

    Returns:
        Path to the final combined comic strip image.
    """
    if len(image_prompts) != len(dialogue_prompts):
        raise ValueError("image_prompts and dialogue_prompts must be the same length.")

    os.makedirs(output_dir, exist_ok=True)
    panel_paths = []

    for i, (prompt, dialogue) in enumerate(zip(image_prompts, dialogue_prompts)):
        output_path = os.path.join(output_dir, f"panel_{i+1}.png")
        generate_scene_with_dialogue(prompt, dialogue, output_path)
        panel_paths.append(output_path)
        print(f"[INFO] Panel {i+1}/{len(image_prompts)} complete.")

    # Stitch panels horizontally
    images = [Image.open(p).convert("RGBA") for p in panel_paths]
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    strip = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 255))
    x_offset = 0
    for img in images:
        strip.paste(img, (x_offset, 0))
        x_offset += img.width

    strip.save(final_output)
    print(f"[INFO] Comic sequence saved: {final_output}")
    return final_output
