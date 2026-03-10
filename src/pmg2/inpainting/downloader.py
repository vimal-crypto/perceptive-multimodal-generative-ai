from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
torch.save(pipe, "stable_diffusion_preloaded.pth")
print("Stable Diffusion model saved as stable_diffusion_preloaded.pth")

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", torch_dtype=torch.float16)
torch.save(model, "dialogpt_preloaded.pth")
print("PersonaGPT model saved as dialogpt_preloaded.pth")
