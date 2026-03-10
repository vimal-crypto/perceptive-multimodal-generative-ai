import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont

device = "cuda" if torch.cuda.is_available() else "cpu"
model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
model = model.to(device)

dialogue_model_name = "gpt2"
dialogue_tokenizer = AutoTokenizer.from_pretrained(dialogue_model_name)
dialogue_model = AutoModelForCausalLM.from_pretrained(dialogue_model_name).to(device)

def generate_dialogue(prompt):
    input_ids = dialogue_tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = dialogue_model.generate(input_ids, max_length=50, num_return_sequences=1,
                                      no_repeat_ngram_size=2, top_p=0.92, temperature=0.7)
    return dialogue_tokenizer.decode(output[0], skip_special_tokens=True)

def generate_scene_with_generated_dialogue(scene_prompt, output_path):
    scene_image = model(scene_prompt).images[0].convert("RGBA")
    dialogue = generate_dialogue(scene_prompt)
    print(f"Generated dialogue: {dialogue}")
    from lat import add_textbox_to_image
    scene_with_text = add_textbox_to_image(scene_image, dialogue)
    scene_with_text.save(output_path)
    print(f"Scene saved as {output_path}")

if __name__ == '__main__':
    scene_prompts = ["Batman is riding his car in a formula 1 race"]
    for i, scene_prompt in enumerate(scene_prompts, start=1):
        generate_scene_with_generated_dialogue(scene_prompt, f"scene_{i}.png")
