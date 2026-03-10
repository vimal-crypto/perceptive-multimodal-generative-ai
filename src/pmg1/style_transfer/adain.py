import torch
import torch.nn as nn

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization for real-time style transfer.
    Normalizes content features using mean and variance from style features.
    AdaIN(x, y) = sigma(y) * ((x - mu(x)) / sigma(x)) + mu(y)
    """
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        assert content.size()[:2] == style.size()[:2], "Batch and channel dims must match"
        size = content.size()
        # Compute content mean and std per channel
        content_mean = content.view(size[0], size[1], -1).mean(dim=2).view(size[0], size[1], 1, 1)
        content_std = content.view(size[0], size[1], -1).std(dim=2).view(size[0], size[1], 1, 1) + 1e-5
        # Compute style mean and std per channel
        style_mean = style.view(size[0], size[1], -1).mean(dim=2).view(size[0], size[1], 1, 1)
        style_std = style.view(size[0], size[1], -1).std(dim=2).view(size[0], size[1], 1, 1) + 1e-5
        # Normalize content and re-scale with style statistics
        normalized = (content - content_mean) / content_std
        return style_std * normalized + style_mean


class AdaINDecoder(nn.Module):
    """Simple decoder that upsamples AdaIN-normalized features back to image space."""
    def __init__(self):
        super(AdaINDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
