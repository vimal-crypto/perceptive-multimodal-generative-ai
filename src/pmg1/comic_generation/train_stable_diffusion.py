import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
from transformers import AdamW


def fine_tune_stable_diffusion(
    dataset_path: str,
    output_dir: str = "./fine_tuned_sd",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 5e-6,
    device: str = None
):
    """
    Fine-tune the Stable Diffusion UNet on a custom image-caption dataset.
    The dataset should return dict keys 'image' and 'caption'.

    Args:
        dataset_path: Hugging Face dataset path or local path.
        output_dir: Directory to save fine-tuned model weights.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate for AdamW optimizer.
        device: 'cuda' or 'cpu'. Auto-detects if None.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Fine-tuning on device: {device}")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    optimizer = AdamW(pipe.unet.parameters(), lr=lr)

    dataset = load_dataset(dataset_path)
    train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    for epoch in range(epochs):
        pipe.unet.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            images = torch.stack([transform(img) for img in batch["image"]]).to(device)
            captions = batch["caption"]
            input_ids = pipe.tokenizer(
                captions, return_tensors="pt", padding=True, truncation=True
            ).input_ids.to(device)

            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            noise_pred = pipe.unet(latents, input_ids).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_dataloader), 1)
        print(f"[Epoch {epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    pipe.unet.save_pretrained(os.path.join(output_dir, "unet"))
    pipe.vae.save_pretrained(os.path.join(output_dir, "vae"))
    pipe.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print(f"[INFO] Fine-tuned model saved to: {output_dir}")
