# PMG-AI Jupyter Notebooks

This directory contains interactive Jupyter notebooks demonstrating the capabilities of the PMG-AI framework.

## Available Notebooks

### PMG-1: Comic Generation and Style Transfer

#### `pmg1_comic_generation_demo.ipynb`
**Description**: Interactive demonstration of comic panel generation using Stable Diffusion

**Features**:
- Persona-driven comic generation
- Multi-panel comic sequences
- Custom character and setting definitions
- Various comic styles (anime, manga, western)
- Comic strip assembly

**Requirements**:
- CUDA-enabled GPU (recommended)
- 8GB+ VRAM
- Stable Diffusion model weights

#### `pmg1_style_transfer_demo.ipynb`
**Description**: Neural style transfer demonstrations

**Features**:
- VGG19-based style transfer
- GAN-based artistic transformation
- AdaIN fast style transfer
- Multiple style examples
- Real-time processing

### PMG-2: Text-to-Image and 3D Reconstruction

#### `pmg2_text_to_image_demo.ipynb`
**Description**: Text-to-image synthesis using CLIP and diffusion models

**Features**:
- CLIP-guided image generation
- Latent diffusion implementation
- VQGAN decoding
- Prompt engineering examples
- Quality comparison

#### `pmg2_2d_to_3d_reconstruction_demo.ipynb`
**Description**: Complete pipeline for converting 2D images to 3D models

**Features**:
- MiDaS depth estimation
- Point cloud generation
- 3D mesh creation
- Visualization with Open3D
- Export to various 3D formats (OBJ, PLY, STL)

**Based on Drive Files**:
- `2d and 3d.ipynb` - Original 2D-to-3D notebook
- `Sidesh_Code.ipynb` - Reference implementation
- Integration with `midas_depth.py` and depth processing code

## Getting Started

### Installation

1. Install Jupyter Notebook or JupyterLab:
```bash
pip install jupyter
# or
pip install jupyterlab
```

2. Install project dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook
# or
jupyter lab
```

### GPU Support

For optimal performance, ensure you have:
- CUDA 11.7 or higher
- cuDNN 8.0 or higher
- NVIDIA GPU with Compute Capability 3.5+

Verify GPU availability:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Device Count: {torch.cuda.device_count()}")
```

## Notebook Structure

Each notebook follows this general structure:

1. **Setup & Imports**
   - Required library imports
   - Environment configuration
   - GPU availability check

2. **Model Initialization**
   - Load pre-trained models
   - Configure parameters
   - Memory optimization

3. **Example Demonstrations**
   - Basic usage examples
   - Advanced techniques
   - Parameter tuning

4. **Results Visualization**
   - Output display
   - Comparison plots
   - Quality metrics

5. **Export & Save**
   - Save generated outputs
   - Export formats
   - File management

## Dataset Requirements

### For Comic Generation:
- No dataset required (uses Stable Diffusion)
- Optional: Custom character reference images

### For Style Transfer:
- Content images (any resolution)
- Style reference images
- Examples provided in `data/examples/`

### For 3D Reconstruction:
- 2D images (JPEG, PNG)
- Recommended: High-resolution (1024x1024+)
- Examples: `data/test_images/`

## Output Locations

Generated outputs are automatically saved to:
- Comics: `outputs/comics/`
- Style transfers: `outputs/style_transfer/`
- Depth maps: `outputs/depth_maps/`
- Point clouds: `outputs/point_clouds/`
- 3D meshes: `outputs/meshes/`

## Tips and Best Practices

### Memory Management
```python
# Clear GPU cache between runs
import torch
torch.cuda.empty_cache()

# Enable memory efficient attention
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()
```

### Batch Processing
```python
# Process multiple images efficiently
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = model.process_batch(images, batch_size=4)
```

### Reproducibility
```python
# Set random seeds for consistent results
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch size
- Enable memory optimizations
- Use smaller model variants
- Clear cache between runs

### Slow Performance
- Verify GPU is being used
- Check CUDA version compatibility
- Update to latest PyTorch version
- Use mixed precision (FP16)

### Model Loading Issues
- Check internet connection (first-time downloads)
- Verify disk space for model weights
- Try manual model download
- Check Hugging Face Hub status

## Contributing

To contribute new notebooks:
1. Follow the existing structure
2. Include comprehensive documentation
3. Add example outputs
4. Test on clean environment
5. Submit pull request

## References

- Stable Diffusion: https://github.com/Stability-AI/stablediffusion
- MiDaS: https://github.com/isl-org/MiDaS
- CLIP: https://github.com/openai/CLIP
- Open3D: http://www.open3d.org/

## Support

For issues or questions:
- GitHub Issues: https://github.com/vimal-crypto/perceptive-multimodal-generative-ai/issues
- Email: contact@example.com

---

**Note**: These notebooks are based on the research papers in the `papers/` directory. Please cite appropriately if using this work in academic publications.
