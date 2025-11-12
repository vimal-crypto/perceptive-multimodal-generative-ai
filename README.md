# Perceptive Multimodal Generative AI (PMG-AI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)

> **A unified multimodal platform for creative AI-driven content generation across text, images, comics, animations, and 3D models.**

PMG-AI is a comprehensive generative AI framework that integrates **deep learning, GANs, NLP, OCR, and diffusion models** to enable end-to-end multimodal content creation, modification, and editing. The platform supports diverse applications including comic generation, image style transfer, text-to-image synthesis, animation, inpainting, and 2D-to-3D reconstruction.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- - [Datasets](#datasets)
- [Usage](#usage)
  - [PMG-1: Comic Generation & Style Transfer](#pmg-1-comic-generation--style-transfer)
  - [PMG-2: Image Generation & 3D Reconstruction](#pmg-2-image-generation--3d-reconstruction)
- [Project Structure](#project-structure)
- [Research Papers](#research-papers)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

Perceptive Multimodal Generative AI (PMG-AI) breaks the boundaries of single-modality systems by providing a holistic platform for creative workflows. The system consists of two major components:

### **PMG-1: Creative Content Generation**
- **Comic Generation**: AI-powered storytelling with Stable Diffusion image synthesis and PersonaGPT dialogue generation
- **Image Style Transfer**: Neural style transfer using VGG19, GANs, and Adaptive Instance Normalization (AdaIN)
- **Text Extraction**: OCR integration for extracting and repurposing textual content from images

### **PMG-2: Advanced Image Synthesis & 3D Modeling**
- **Text-to-Image Generation**: CLIP-based encoding with Latent Diffusion Models (LDM) for semantic image synthesis
- **Animated Content**: Multi-frame generation with temporal smoothness for GIF/MP4 outputs
- **Masked Inpainting**: Selective region-specific image editing using Stable Diffusion
- **2D-to-3D Reconstruction**: Voxel-based modeling, depth estimation, and mesh extraction for AR/VR applications

**Developed by:** Rithani M., Sidesh Sundar S., Vimal Dharan N., SyamDev R. S.  
**Institution:** Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai

---

## ✨ Key Features

### **Multimodal Content Creation**
- 🎨 **Comic Generation**: Fully customizable comics with narrative and visual synthesis
- 🖼️ **Image Style Transfer**: Apply artistic styles while preserving content structure
- 📝 **Text-to-Image**: Generate high-quality images from natural language descriptions
- 🎬 **Animation**: Create animated sequences from static prompts
- ✂️ **Inpainting**: Edit specific regions without affecting the entire image
- 🧊 **2D-to-3D Conversion**: Transform 2D images into 3D models with depth estimation

### **Advanced AI Technologies**
- **CLIP Encoder**: Semantic understanding of text prompts
- **Latent Diffusion Models (LDM)**: Efficient image generation in latent space
- **GANs**: Generative Adversarial Networks for realistic style transfer
- **VGG19**: Deep feature extraction for content and style preservation
- **PersonaGPT**: Character-consistent dialogue generation
- **Stable Diffusion**: State-of-the-art denoising for image synthesis
- **Voxel Modeling**: 3D reconstruction with Marching Cubes algorithm

### **Scalability & Flexibility**
- **Modular Architecture**: Independent PMG-1 and PMG-2 modules
- **Customizable Workflows**: Tailor AI models to specific artistic preferences
- **Cross-Platform**: Supports Windows, Linux, and macOS
- **Multiple Output Formats**: PNG, JPEG, GIF, MP4, OBJ, PLY

---

## 🏛️ System Architecture

```
PMG-AI Framework
│
├── PMG-1: Creative Content Generation
│   ├── Comic Generation
│   │   ├── Stable Diffusion (Image Synthesis)
│   │   └── PersonaGPT (Dialogue Generation)
│   ├── Image Style Transfer
│   │   ├── VGG19 (Feature Extraction)
│   │   ├── GAN (Adversarial Training)
│   │   └── AdaIN (Style Normalization)
│   └── Text Extraction (OCR)
│
└── PMG-2: Image Synthesis & 3D Reconstruction
    ├── Text-to-Image Generation
    │   ├── CLIP Encoder (Text → Embedding)
    │   ├── Latent Diffusion Model (LDM)
    │   └── VQGAN Decoder (Latent → Image)
    ├── Animation Generation
    │   └── Multi-frame Synthesis + Temporal Smoothing
    ├── Masked Inpainting
    │   └── Stable Diffusion + Binary Masks
    └── 2D-to-3D Reconstruction
        ├── Depth Estimation (MiDaS)
        ├── Voxel Grid Generation
        ├── Keypoint Estimation
        └── Mesh Extraction (Marching Cubes)
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU (recommended for faster processing)
- 16GB RAM minimum (32GB recommended)

### Clone the Repository
```bash
git clone https://github.com/vimal-crypto/perceptive-multimodal-generative-ai.git
cd perceptive-multimodal-generative-ai
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Pre-trained Models
```bash
# Stable Diffusion weights
wget https://huggingface.co/stabilityai/stable-diffusion-2-1

# VGG19 weights (auto-downloaded by PyTorch)
# CLIP weights (auto-downloaded)
```

---

## 🚀 Usage

### PMG-1: Comic Generation & Style Transfer

#### **Comic Generation**
```python
from src.pmg1.comic_generation import ComicGenerator

generator = ComicGenerator()
comic_panels = generator.generate_comic(
    narrative="A superhero flying over a city during sunset",
    num_panels=4,
    style="anime"
)
generator.save_comic(comic_panels, "outputs/comic_output.png")
```

#### **Image Style Transfer**
```python
from src.pmg1.style_transfer import StyleTransfer

transfer = StyleTransfer()
stylized_image = transfer.apply_style(
    content_image="inputs/photo.jpg",
    style_image="inputs/style_art.jpg"
)
transfer.save(stylized_image, "outputs/stylized_photo.jpg")
```

### PMG-2: Image Generation & 3D Reconstruction

#### **Text-to-Image Generation**
```python
from src.pmg2.text_to_image import TextToImageGenerator

generator = TextToImageGenerator()
image = generator.generate(
    prompt="A futuristic city with neon lights and flying cars",
    height=512,
    width=512,
    num_inference_steps=50
)
generator.save(image, "outputs/generated_city.png")
```

#### **Animated Content**
```python
from src.pmg2.animation import AnimationGenerator

animator = AnimationGenerator()
animation = animator.generate_animation(
    prompt="A forest fire spreading across hills",
    num_frames=30,
    fps=10
)
animator.save(animation, "outputs/forest_fire.gif")
```

#### **Masked Inpainting**
```python
from src.pmg2.inpainting import InpaintingModel

inpainter = InpaintingModel()
edited_image = inpainter.inpaint(
    image="inputs/car.jpg",
    mask="inputs/mask.png",
    prompt="Replace with a red sports car"
)
inpainter.save(edited_image, "outputs/inpainted_car.jpg")
```

#### **2D-to-3D Reconstruction**
```python
from src.pmg2.reconstruction_2d_3d import Reconstruction2Dto3D

reconstructor = Reconstruction2Dto3D()
mesh = reconstructor.convert(
    image="inputs/car.jpg",
    output_format="obj"
)
reconstructor.save_mesh(mesh, "outputs/car_3d.obj")
```

---

## 📁 Project Structure

```
perceptive-multimodal-generative-ai/
├── src/
│   ├── pmg1/                      # PMG-1 Modules
│   │   ├── comic_generation/
│   │   │   ├── stable_diffusion.py   # Image synthesis for comic panels
│   │   │   └── persona_gpt.py        # Dialogue generation
│   │   ├── style_transfer/
│   │   │   ├── vgg19_extractor.py    # Feature extraction
│   │   │   ├── gan_transfer.py       # GAN-based transfer
│   │   │   └── adain.py              # Adaptive Instance Normalization
│   │   └── text_extraction/
│   │       └── ocr_extractor.py      # OCR for text extraction
│   │
│   ├── pmg2/                      # PMG-2 Modules
│   │   ├── text_to_image/
│   │   │   ├── clip_encoder.py       # CLIP-based text encoding
│   │   │   ├── latent_diffusion.py   # LDM for image synthesis
│   │   │   └── vqgan_decoder.py      # VQGAN decoder
│   │   ├── animation/
│   │   │   └── animation_generator.py # Multi-frame animation
│   │   ├── inpainting/
│   │   │   └── inpainting_model.py   # Masked region editing
│   │   └── reconstruction_2d_3d/
│   │       ├── depth_estimation.py   # MiDaS depth maps
│   │       ├── voxel_modeling.py     # 3D voxel grids
│   │       ├── keypoint_estimator.py # 2D keypoint detection
│   │       └── mesh_generator.py     # Marching Cubes mesh extraction
│   │
│   └── utils/                     # Utility functions
│       ├── image_processing.py
│       ├── file_io.py
│       └── visualization.py
│
├── notebooks/                     # Jupyter notebooks for experiments
│   ├── comic_generation_demo.ipynb
│   ├── style_transfer_demo.ipynb
│   ├── text_to_image_demo.ipynb
│   └── 2d_to_3d_reconstruction.ipynb
│
├── outputs/                       # Generated outputs
│   ├── comics/
│   ├── style_transfer/
│   ├── text_to_image/
│   ├── animations/
│   └── 3d_models/
│
├── papers/                        # Research papers
│   ├── PMG_AI.pdf                # PMG-1 research paper
│   └── PMG_AI_2.pdf              # PMG-2 research paper
│
├── docs/                          # Documentation
│   ├── installation.md
│   ├── usage_guide.md
│   ├── api_reference.md
│   └── troubleshooting.md
│
├── assets/                        # Images and media
│   ├── examples/
│   └── screenshots/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── .gitignore                     # Git ignore file
```

---

## 📚 Research Papers

This project is based on two comprehensive research papers:

### **Paper 1: Perceptive Multimodal Generative AI (PMG-1)**
**Authors:** Rithani M., Sidesh Sundar S., Vimal Dharan N., SyamDev R. S.  
**Focus:** Comic generation, image style transfer, text extraction, and template customization

**Key Contributions:**
- Novel comic generation framework using Stable Diffusion + PersonaGPT
- Advanced style transfer with VGG19 and AdaIN
- Comprehensive feature extraction (MAV, ZCR, RMS)
- OCR integration for text repurposing

**[Read Paper](papers/PMG_AI.pdf)**

### **Paper 2: Perceptive Multimodal Generative AI (PMG-2)**
**Authors:** Rithani M., Sidesh Sundar S., Vimal Dharan N.  
**Focus:** Text-to-image synthesis, animation, inpainting, and 2D-to-3D reconstruction

**Key Contributions:**
- CLIP-based latent diffusion for text-to-image generation
- Temporal smoothness for animated content
- Mask-based inpainting with semantic conditioning
- Voxel-based 3D reconstruction with depth estimation

**[Read Paper](papers/PMG_AI_2.pdf)**

---

## 🏆 Results

### PMG-1 Results

#### Comic Generation
- **Accuracy**: 94% semantic alignment between text prompts and generated images
- **Dialogue Coherence**: 92% character consistency across panels
- **Generation Time**: ~5 seconds per panel (GPU accelerated)

#### Image Style Transfer
- **Content Preservation**: 98% structural similarity (SSIM)
- **Style Accuracy**: 95% style matching with target artwork
- **Edge Preservation**: Edge detection maps show 97% feature retention

### PMG-2 Results

#### Text-to-Image Generation
- **Semantic Accuracy**: 96% alignment with text prompts (CLIP score)
- **Image Quality**: FID score of 12.5 (lower is better)
- **Generation Time**: ~8 seconds for 512x512 images

#### 2D-to-3D Reconstruction
- **Depth Map Accuracy**: 91% correlation with ground truth
- **Mesh Quality**: IoU of 0.87 for voxel-based models
- **Processing Time**: ~15 seconds per image

---

## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <b>Rithani M.</b><br>
      <sub>Lead Researcher</sub>
    </td>
    <td align="center">
      <b>Sidesh Sundar S.</b><br>
      <sub>ML Engineer</sub>
    </td>
    <td align="center">
      <b>Vimal Dharan N.</b><br>
      <sub>Deep Learning Specialist</sub>
    </td>
    <td align="center">
      <b>SyamDev R. S.</b><br>
      <sub>AI Researcher</sub>
    </td>
  </tr>
</table>

**Institution:** Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai, India

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Vimal Dharan, Rithani M., Sidesh Sundar S.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

- **Amrita Vishwa Vidyapeetham** for research support and infrastructure
- **Stability AI** for Stable Diffusion models
- **OpenAI** for CLIP and GPT architectures
- **Hugging Face** for model hosting and community support
- Open-source contributors for PyTorch, TensorFlow, and related libraries

---

## 📞 Contact

For questions, collaborations, or feedback:

- **Vimal Dharan**: vimalvimal1293@gmail.com
- **GitHub**: [@vimal-crypto](https://github.com/vimal-crypto)
- **Project Repository**: [perceptive-multimodal-generative-ai](https://github.com/vimal-crypto/perceptive-multimodal-generative-ai)

---

## ⭐ Star This Repository

If you find PMG-AI useful for your research or projects, please consider giving it a star! ⭐

---


---

## 📊 Datasets

PMG-AI utilizes several datasets for training and evaluation across its two main components.

### PMG-1: Text and Image Generation Datasets

#### LAION-400M
- **Purpose**: Text-to-image synthesis and comic generation
- **Size**: 400 million image-text pairs
- **Format**: Image URLs with alt-text captions
- **Source**: Large-scale web-crawled dataset
- **Usage**: Training Stable Diffusion models for comic panel generation
- **License**: Creative Commons (varies by image)
- **Access**: [LAION-400M Official](https://laion.ai/blog/laion-400-open-dataset/)

#### Custom Comic Dataset
- **Purpose**: Comic generation and style transfer
- **Size**: 10,000+ comic panels
- **Format**: JPG/PNG images with dialogue annotations
- **Features**: Character bounding boxes, speech bubble locations, text content
- **Usage**: Fine-tuning models for comic-specific generation

#### Persona Dataset
- **Purpose**: Character-driven dialogue generation
- **Size**: 5,000+ character profiles
- **Format**: JSON files with character attributes
- **Features**: Personality traits, speech patterns, visual descriptions
- **Usage**: PersonaGPT dialogue generation

### PMG-2: 3D Reconstruction Datasets

#### Pix3D Dataset
- **Purpose**: 2D-to-3D reconstruction training
- **Size**: 10,069 image-3D pairs
- **Categories**: Furniture, household objects
- **Format**: 
  - Images: 256x256 RGB
  - 3D models: OBJ format with textures
  - Annotations: Camera parameters, object masks
- **Source**: [Pix3D Official](http://pix3d.csail.mit.edu/)
- **Usage**: Training depth estimation and 3D reconstruction models
- **License**: MIT License

#### ShapeNet Core
- **Purpose**: 3D shape understanding and reconstruction
- **Size**: 51,300+ unique 3D models
- **Categories**: 55 common object categories
- **Format**: OBJ, PLY mesh files
- **Source**: [ShapeNet Official](https://shapenet.org/)
- **Usage**: Pre-training NeRF and mesh reconstruction models
- **License**: ShapeNet Terms of Use

#### Custom Depth Dataset
- **Purpose**: Depth estimation fine-tuning
- **Size**: 2,000+ indoor/outdoor scenes
- **Format**: 
  - RGB images: 512x512
  - Depth maps: 16-bit PNG
  - Point clouds: PLY format
- **Source**: Captured using Intel RealSense D435
- **Usage**: Fine-tuning MiDaS depth estimation

### Data Organization

Datasets should be organized in the following structure:

```
data/
├── pmg1/
│   ├── comics/
│   │   ├── images/
│   │   └── annotations/
│   ├── personas/
│   └── laion_subset/
└── pmg2/
    ├── pix3d/
    │   ├── img/
    │   ├── mask/
    │   └── model/
    ├── shapenet/
    └── depth_custom/
        ├── rgb/
        ├── depth/
        └── pointclouds/
```

### Downloading Datasets

**LAION-400M** (subset):
```bash
wget https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/
```

**Pix3D**:
```bash
wget http://pix3d.csail.mit.edu/data/pix3d.zip
unzip pix3d.zip -d data/pmg2/
```

**ShapeNet**:
1. Register at https://shapenet.org/
2. Download ShapeNetCore.v2
3. Extract to `data/pmg2/shapenet/`

### Dataset Statistics

| Dataset | Size | Type | Resolution | Format |
|---------|------|------|------------|--------|
| LAION-400M | 400M pairs | Text-Image | Varies | JPG/PNG |
| Comic Custom | 10K panels | Image + Text | 512x512 | PNG |
| Pix3D | 10K pairs | RGB + 3D | 256x256 | JPG + OBJ |
| ShapeNet | 51K models | 3D Mesh | N/A | OBJ/PLY |
| Depth Custom | 2K scenes | RGB + Depth | 512x512 | PNG + PLY |

### Data Preprocessing

All datasets undergo preprocessing before training:

1. **Image Normalization**: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
2. **Resizing**: Images resized to 512x512 for PMG-1, 256x256 for PMG-2
3. **Augmentation**: Random flips, rotations, color jittering
4. **Depth Map Normalization**: Min-max scaling to [0, 1] range

For detailed preprocessing scripts, see `src/data_processing/`.


**Made with ❤️ by the PMG-AI Team @ Amrita School of Computing**
