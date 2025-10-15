# CLP-RGBD-ViT

## Qwen3-VL Vision Transformer Implementation

This repository contains a PyTorch implementation of the Vision Transformer (ViT) architecture used in Qwen3-VL, along with a complete pipeline for processing images and extracting visual tokens.

## Overview

Qwen3-VL is a multimodal large language model that uses a Vision Transformer to process images. This implementation provides:

- **Complete ViT Architecture**: Full implementation of the vision transformer with all key components
- **Rotary Position Embedding (RoPE)**: 2D spatial position encoding for patches
- **Image-to-Token Pipeline**: End-to-end pipeline from raw images to visual tokens
- **Multiple Model Sizes**: Support for small, base, and large model configurations
- **Flexible Processing**: Support for single images, batches, and different pooling strategies

## Key Features

### 1. Vision Transformer Architecture

The Qwen3-VL ViT includes:

- **Patch Embedding**: Converts images into patch tokens (default: 14x14 patches)
- **Rotary Position Embedding (RoPE)**: Encodes spatial relationships between patches
- **Multi-Head Self-Attention**: Captures relationships between different image regions
- **Feed-Forward Networks**: Processes attention outputs
- **Layer Normalization**: Stabilizes training and inference
- **Residual Connections**: Enables deep network training

### 2. Image Processing Pipeline

Complete pipeline stages:

1. **Image Preprocessing**:
   - Resize to target size (default: 224x224)
   - Normalize using ImageNet statistics
   - Convert to tensor format

2. **Vision Transformer**:
   - Patch embedding: Image → Patches
   - Position encoding: Add spatial information
   - Transformer layers: Extract visual features
   - Output: Visual token embeddings

3. **Token Post-processing**:
   - Optional pooling strategies (mean, max, cls, or none)
   - Format conversion (tensor, numpy, list)

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- NumPy >= 1.24.0
- Pillow >= 9.0.0

Optional:
- matplotlib >= 3.5.0 (for visualization)

## Usage

### Quick Start

```python
from image_to_token_pipeline import Qwen3VLPipeline
import numpy as np

# Create pipeline
pipeline = Qwen3VLPipeline(model_size="base")

# Process an image (can be path, PIL Image, or numpy array)
image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
tokens = pipeline.process_image(image)

print(f"Output shape: {tokens.shape}")  # (1, 256, 1024)
# 1 = batch size
# 256 = number of patches (16x16 grid for 224x224 image with 14x14 patches)
# 1024 = embedding dimension
```

### Advanced Usage

#### 1. Different Model Sizes

```python
from image_to_token_pipeline import Qwen3VLPipeline

# Small model (faster, less parameters)
pipeline_small = Qwen3VLPipeline(model_size="small")

# Base model (balanced)
pipeline_base = Qwen3VLPipeline(model_size="base")

# Large model (more capacity)
pipeline_large = Qwen3VLPipeline(model_size="large")
```

#### 2. Batch Processing

```python
from image_to_token_pipeline import Qwen3VLPipeline

pipeline = Qwen3VLPipeline(model_size="base")

# Process multiple images at once
images = [image1, image2, image3]  # List of images
tokens = pipeline.process_batch(images)

print(f"Batch output shape: {tokens.shape}")  # (3, 256, 1024)
```

#### 3. Different Pooling Strategies

```python
from image_to_token_pipeline import Qwen3VLPipeline

# No pooling - keep all tokens (default)
pipeline = Qwen3VLPipeline(pooling_strategy="none")
tokens = pipeline.process_image(image)  # Shape: (1, 256, 1024)

# Mean pooling - average across spatial dimension
pipeline = Qwen3VLPipeline(pooling_strategy="mean")
tokens = pipeline.process_image(image)  # Shape: (1, 1, 1024)

# Max pooling - max across spatial dimension
pipeline = Qwen3VLPipeline(pooling_strategy="max")
tokens = pipeline.process_image(image)  # Shape: (1, 1, 1024)
```

#### 4. Using the Model Directly

```python
from qwen3_vl_vit import create_qwen3_vl_vit
from image_to_token_pipeline import ImagePreprocessor
import torch

# Create model
model = create_qwen3_vl_vit(model_size="base", img_size=224, patch_size=14)
model.eval()

# Preprocess image
preprocessor = ImagePreprocessor(img_size=224)
image_tensor = preprocessor.preprocess("path/to/image.jpg")

# Extract tokens
with torch.no_grad():
    tokens = model(image_tensor)

print(f"Tokens shape: {tokens.shape}")
```

#### 5. Custom Configuration

```python
from qwen3_vl_vit import Qwen3VisionTransformer

# Create custom ViT configuration
model = Qwen3VisionTransformer(
    img_size=224,           # Input image size
    patch_size=14,          # Patch size
    in_channels=3,          # RGB channels
    embed_dim=768,          # Embedding dimension
    depth=12,               # Number of transformer layers
    num_heads=12,           # Number of attention heads
    mlp_ratio=4.0,          # MLP hidden dim ratio
    dropout=0.1,            # Dropout rate
)
```

## Architecture Details

### Model Configurations

| Model Size | Embed Dim | Depth | Heads | Parameters |
|------------|-----------|-------|-------|------------|
| Small      | 512       | 12    | 8     | ~22M       |
| Base       | 1024      | 24    | 16    | ~86M       |
| Large      | 1280      | 32    | 16    | ~304M      |

### Patch Embedding

- Default: 224×224 image → 16×16 grid of 14×14 patches → 256 tokens
- Each patch is linearly projected to embedding dimension
- Spatial relationships preserved through grid structure

### Rotary Position Embedding (RoPE)

- Extends 1D RoPE to 2D spatial grids
- Encodes relative spatial positions
- Applied in attention mechanism (query and key)
- Better generalization to different image sizes

### Transformer Layers

Each layer contains:
1. Layer normalization
2. Multi-head self-attention with RoPE
3. Residual connection
4. Layer normalization
5. Feed-forward network (MLP)
6. Residual connection

## Pipeline Flow

```
Input Image (H×W×3)
    ↓
[Resize & Normalize]
    ↓
Image Tensor (3×224×224)
    ↓
[Patch Embedding]
    ↓
Patch Tokens (256×D)
    ↓
[Add Position Encoding - RoPE]
    ↓
[Transformer Layers] × N
    ↓
Visual Tokens (256×D)
    ↓
[Optional Pooling]
    ↓
Output Tokens
```

Where:
- H, W: Original image height and width
- D: Embedding dimension (512/1024/1280 depending on model size)
- N: Number of transformer layers (12/24/32 depending on model size)

## Examples

Run the provided examples to see different use cases:

```bash
python examples.py
```

This will demonstrate:
1. Basic usage with dummy images
2. Batch processing
3. Different pooling strategies
4. Model architecture details
5. Step-by-step processing
6. Pipeline configuration
7. NumPy output format

## File Structure

```
.
├── qwen3_vl_vit.py              # ViT model implementation
├── image_to_token_pipeline.py    # Complete processing pipeline
├── examples.py                   # Usage examples
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Implementation Details

### Key Components

1. **PatchEmbed**: Converts images to patch embeddings using Conv2d
2. **RotaryPositionEmbedding**: Generates and applies RoPE to Q and K
3. **MultiHeadAttention**: Self-attention with RoPE support
4. **MLP**: Two-layer feed-forward network with GELU
5. **TransformerBlock**: Complete transformer encoder block
6. **Qwen3VisionTransformer**: Full ViT architecture

### Design Decisions

- **RoPE over Absolute PE**: Better generalization and relative position modeling
- **Pre-norm Architecture**: Layer norm before attention/MLP (more stable)
- **GELU Activation**: Smoother than ReLU, better for transformers
- **No CLS Token**: Uses all patch tokens (no special classification token)
- **Flexible Output**: Support for different pooling strategies

## Technical Notes

### Memory Requirements

Approximate GPU memory for batch size 1:

| Model Size | Image Size | Memory  |
|------------|------------|---------|
| Small      | 224×224    | ~1 GB   |
| Base       | 224×224    | ~2 GB   |
| Large      | 224×224    | ~4 GB   |

### Performance Tips

1. **Batch Processing**: Process multiple images together for better GPU utilization
2. **Mixed Precision**: Use `torch.cuda.amp` for faster inference
3. **Model Caching**: Keep model loaded for multiple inferences
4. **Image Preprocessing**: Preprocess images in parallel if processing many

### Extensibility

The implementation is modular and can be extended:

- Add temporal dimension for video processing
- Integrate with language models for multimodal tasks
- Fine-tune on specific datasets
- Add different position encoding schemes
- Modify attention mechanisms

## Qwen3-VL Specifics

This implementation follows Qwen3-VL's design:

1. **Patch Size**: 14×14 (creates 256 tokens for 224×224 images)
2. **Position Encoding**: RoPE for 2D spatial awareness
3. **Architecture**: Standard ViT with pre-norm
4. **No CLS Token**: All patch tokens used
5. **Output**: Dense visual tokens for language model

The tokens can be directly fed into Qwen3's language model after appropriate projection.

## Citation

If you use this implementation, please cite:

```bibtex
@software{qwen3_vl_vit_implementation,
  title = {Qwen3-VL Vision Transformer Implementation},
  author = {CLP-RGBD-ViT Contributors},
  year = {2024},
  url = {https://github.com/X-Green/CLP-RGBD-ViT}
}
```

For Qwen3-VL model:
```bibtex
@article{qwen3vl,
  title={Qwen3-VL: A Multimodal Large Language Model},
  author={Qwen Team},
  year={2024}
}
```

## License

This implementation is provided for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Qwen Team for the Qwen3-VL architecture
- Vision Transformer (ViT) original paper: "An Image is Worth 16x16 Words"
- RoPE paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"