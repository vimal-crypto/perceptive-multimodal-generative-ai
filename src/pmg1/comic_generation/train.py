import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
from transformers import AdamW

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cpu")

optimizer = AdamW(pipe.unet.parameters(), lr=5e-6)

dataset = load_dataset("path_to_your_dataset")
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

epochs = 5
for epoch in range(epochs):
    pipe.unet.train()
    for batch in train_dataloader:
        images = batch["image"].to("cpu")
        captions = batch["caption"]
        images = transform(images).unsqueeze(0)
        input_ids = pipe.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids
        with torch.no_grad():
            latents = pipe.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
        noise = torch.randn_like(latents)
        noise_pred = pipe.unet(latents, input_ids).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

pipe.unet.save_pretrained("./fine_tuned_stable_diffusion_unet")
pipe.vae.save_pretrained("./fine_tuned_stable_diffusion_vae")
pipe.tokenizer.save_pretrained("./fine_tuned_stable_diffusion_tokenizer")
