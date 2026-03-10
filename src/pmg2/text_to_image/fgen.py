import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline as hf_pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
sd_pipe = sd_pipe.to(device)

sentiment_analyzer = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text: str) -> str:
    result = sentiment_analyzer(text)[0]
    return result['label']

def generate_image_from_text(prompt: str, output_path: str = "generated_image.png") -> str:
    sentiment = analyze_sentiment(prompt)
    style_prefix = "vibrant, colorful" if sentiment == "POSITIVE" else "dark, moody"
    enhanced_prompt = f"{style_prefix}, {prompt}"
    image = sd_pipe(enhanced_prompt).images[0]
    image.save(output_path)
    print(f"[INFO] Image saved to {output_path}")
    return output_path

if __name__ == '__main__':
    generate_image_from_text("A futuristic city with flying cars")
