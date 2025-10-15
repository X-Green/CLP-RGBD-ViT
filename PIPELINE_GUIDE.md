# Qwen3-VL Vision Transformer Pipeline Guide

## Quick Start Guide

This guide provides a quick overview of how to use the Qwen3-VL ViT implementation to process images and generate visual tokens.

## Pipeline Overview

The complete pipeline consists of three main stages:

```
Image Input → Preprocessing → Vision Transformer → Visual Tokens Output
```

## Stage 1: Image Input

### Supported Input Formats

The pipeline accepts three types of image inputs:

1. **File Path** (string)
   ```python
   image = "path/to/image.jpg"
   ```

2. **PIL Image**
   ```python
   from PIL import Image
   image = Image.open("path/to/image.jpg")
   ```

3. **NumPy Array**
   ```python
   import numpy as np
   image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
   ```

## Stage 2: Preprocessing

The preprocessing stage transforms the input image into a format suitable for the ViT:

### Operations:
1. **Resize**: Image resized to 224×224 (bicubic interpolation)
2. **Convert**: Convert to PyTorch tensor
3. **Normalize**: Apply ImageNet normalization
   - Mean: (0.485, 0.456, 0.406)
   - Std: (0.229, 0.224, 0.225)

### Output:
- Tensor of shape: `(1, 3, 224, 224)`
- Format: `(batch, channels, height, width)`
- Values: Normalized float32 values

## Stage 3: Vision Transformer

The ViT processes the preprocessed image through multiple stages:

### 3.1 Patch Embedding
- Divides 224×224 image into 16×16 grid
- Each cell is a 14×14 patch
- Total: 256 patches
- Each patch → embedding vector

### 3.2 Position Encoding
- Adds spatial position information
- Uses Rotary Position Embedding (RoPE)
- Encodes 2D grid structure

### 3.3 Transformer Layers
- Multiple layers (12/24/32 depending on model size)
- Each layer contains:
  * Multi-head self-attention
  * Feed-forward network
  * Layer normalization
  * Residual connections

### 3.4 Output Generation
- Final layer normalization
- Optional pooling
- Visual token embeddings

## Stage 4: Visual Tokens Output

### Output Format

The pipeline produces visual token embeddings:

```
Shape: (batch_size, num_tokens, embed_dim)
```

**Example for single image with base model:**
- Shape: `(1, 256, 1024)`
- `1`: Batch size (single image)
- `256`: Number of visual tokens (one per patch)
- `1024`: Embedding dimension

### Token Meanings

Each of the 256 tokens represents a 14×14 region of the original image:

```
Token Grid (16×16):
┌─────┬─────┬─────┬───────┬─────┐
│ T0  │ T1  │ T2  │  ...  │ T15 │  Row 0
├─────┼─────┼─────┼───────┼─────┤
│ T16 │ T17 │ T18 │  ...  │ T31 │  Row 1
├─────┼─────┼─────┼───────┼─────┤
│ T32 │ T33 │ T34 │  ...  │ T47 │  Row 2
├─────┼─────┼─────┼───────┼─────┤
│ ... │ ... │ ... │  ...  │ ... │
├─────┼─────┼─────┼───────┼─────┤
│T240 │T241 │T242 │  ...  │T255 │  Row 15
└─────┴─────┴─────┴───────┴─────┘

Each token: 1024-dimensional vector (base model)
```

## Complete Usage Examples

### Example 1: Basic Single Image Processing

```python
from image_to_token_pipeline import Qwen3VLPipeline
import numpy as np

# Create pipeline
pipeline = Qwen3VLPipeline(model_size="base")

# Create or load image
image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# Process image
tokens = pipeline.process_image(image)

print(f"Output shape: {tokens.shape}")  # (1, 256, 1024)
```

### Example 2: Batch Processing

```python
from image_to_token_pipeline import Qwen3VLPipeline

pipeline = Qwen3VLPipeline(model_size="base")

# Process multiple images
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
tokens = pipeline.process_batch(images)

print(f"Output shape: {tokens.shape}")  # (3, 256, 1024)
```

### Example 3: With Different Pooling

```python
from image_to_token_pipeline import Qwen3VLPipeline

# No pooling - keep all tokens (default)
pipeline_all = Qwen3VLPipeline(pooling_strategy="none")
tokens_all = pipeline_all.process_image(image)
# Shape: (1, 256, 1024)

# Mean pooling - single token
pipeline_mean = Qwen3VLPipeline(pooling_strategy="mean")
tokens_mean = pipeline_mean.process_image(image)
# Shape: (1, 1, 1024)

# Max pooling - single token
pipeline_max = Qwen3VLPipeline(pooling_strategy="max")
tokens_max = pipeline_max.process_image(image)
# Shape: (1, 1, 1024)
```

### Example 4: Step-by-Step Processing

```python
from qwen3_vl_vit import create_qwen3_vl_vit
from image_to_token_pipeline import ImagePreprocessor
import torch

# Step 1: Create model
model = create_qwen3_vl_vit(model_size="base")
model.eval()

# Step 2: Preprocess image
preprocessor = ImagePreprocessor(img_size=224)
image_tensor = preprocessor.preprocess("image.jpg")

# Step 3: Extract tokens
with torch.no_grad():
    tokens = model(image_tensor)

print(f"Tokens shape: {tokens.shape}")  # (1, 256, 1024)
```

## Understanding the Output

### Token Embeddings

Each token is a 1024-dimensional vector (for base model) that represents:
- Visual features of a specific image region
- Contextual information from surrounding regions
- High-level semantic information

### What You Can Do With Tokens

1. **Feed to Language Model**
   ```python
   # Project tokens to language model dimension
   projection = nn.Linear(1024, 4096)
   lm_tokens = projection(tokens)
   
   # Concatenate with text tokens
   combined = torch.cat([text_tokens, lm_tokens], dim=1)
   
   # Feed to language model
   output = language_model(combined)
   ```

2. **Visual Analysis**
   ```python
   # Compute token statistics
   token_norms = tokens.norm(dim=-1)  # L2 norm per token
   token_mean = tokens.mean(dim=1)    # Average across tokens
   
   # Find most informative tokens
   importance = token_norms.squeeze()
   top_tokens = importance.topk(10).indices
   ```

3. **Visualization**
   ```python
   from image_to_token_pipeline import visualize_tokens
   
   visualize_tokens(
       tokens, 
       save_path="token_visualization.png",
       title="Visual Tokens"
   )
   ```

## Model Configurations Comparison

| Aspect           | Small        | Base          | Large         |
|------------------|--------------|---------------|---------------|
| Embed Dim        | 512          | 1024          | 1280          |
| Layers           | 12           | 24            | 32            |
| Attention Heads  | 8            | 16            | 16            |
| Parameters       | ~38M         | ~303M         | ~630M         |
| Output Shape     | (B, 256, 512)| (B, 256, 1024)| (B, 256, 1280)|
| Speed            | Fast         | Medium        | Slow          |
| Quality          | Good         | Better        | Best          |
| Memory (BS=1)    | ~1 GB        | ~2 GB         | ~4 GB         |

## Performance Tips

### 1. Use Appropriate Model Size
- **Small**: Quick experiments, prototyping, limited compute
- **Base**: Production use, balanced performance
- **Large**: Research, when accuracy is critical

### 2. Batch Processing
```python
# More efficient
tokens = pipeline.process_batch([img1, img2, img3, img4])

# Less efficient
tokens = [pipeline.process_image(img) for img in images]
```

### 3. GPU Acceleration
```python
# Use GPU if available
pipeline = Qwen3VLPipeline(
    model_size="base",
    device="cuda"  # or "cpu"
)
```

### 4. Preprocessing Outside Loop
```python
from image_to_token_pipeline import ImagePreprocessor

preprocessor = ImagePreprocessor()

# Preprocess all images first
tensors = [preprocessor.preprocess(img) for img in images]

# Then batch process
batch = torch.cat(tensors)
tokens = model(batch)
```

## Common Issues and Solutions

### Issue 1: Wrong Image Size
```python
# Error: Image size doesn't match
# Solution: The preprocessor handles resizing automatically
image = Image.open("any_size.jpg")  # Can be any size
tokens = pipeline.process_image(image)  # Automatically resized
```

### Issue 2: Wrong Input Format
```python
# Error: Tensor has wrong shape
# Solution: Use preprocessor
from image_to_token_pipeline import ImagePreprocessor

preprocessor = ImagePreprocessor()
correct_tensor = preprocessor.preprocess(image)
```

### Issue 3: Out of Memory
```python
# Solution 1: Use smaller model
pipeline = Qwen3VLPipeline(model_size="small")

# Solution 2: Reduce batch size
tokens = pipeline.process_batch(images[:4])  # Process in smaller batches

# Solution 3: Use CPU
pipeline = Qwen3VLPipeline(device="cpu")
```

## Integration Example: Multimodal System

Here's how to integrate the ViT with a language model:

```python
import torch
import torch.nn as nn
from image_to_token_pipeline import Qwen3VLPipeline

class MultimodalModel(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.vision_model = Qwen3VLPipeline(model_size="base")
        self.projection = nn.Linear(1024, 4096)  # Project to LM dim
        self.language_model = language_model
    
    def forward(self, image, text_tokens):
        # Extract visual tokens
        visual_tokens = self.vision_model.process_image(image)
        
        # Project to language model dimension
        visual_tokens = self.projection(visual_tokens)
        
        # Combine with text tokens
        combined_tokens = torch.cat([visual_tokens, text_tokens], dim=1)
        
        # Generate response
        output = self.language_model(combined_tokens)
        
        return output
```

## Next Steps

1. **Try the Examples**
   ```bash
   python examples.py
   ```

2. **Run Tests**
   ```bash
   python test_implementation.py
   ```

3. **Read Detailed Architecture**
   - See `ARCHITECTURE.md` for technical details
   - See `README.md` for full documentation

4. **Experiment**
   - Try different model sizes
   - Test with your own images
   - Integrate with your application

## Summary

The Qwen3-VL ViT pipeline provides a simple yet powerful way to convert images into visual tokens:

```python
# Three lines to get visual tokens:
from image_to_token_pipeline import Qwen3VLPipeline
pipeline = Qwen3VLPipeline(model_size="base")
tokens = pipeline.process_image("image.jpg")
```

The output tokens can be used for:
- Multimodal language models
- Visual question answering
- Image captioning
- Visual reasoning
- And more!
