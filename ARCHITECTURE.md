# Qwen3-VL Vision Transformer Architecture

## Overview

This document provides a detailed explanation of the Qwen3-VL Vision Transformer architecture and the image-to-token pipeline.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                               │
│                   (H × W × 3 RGB)                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              IMAGE PREPROCESSING                             │
│  • Resize to 224×224                                         │
│  • Convert to Tensor                                         │
│  • Normalize (ImageNet stats)                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              PATCH EMBEDDING                                 │
│  Conv2d(3, embed_dim, kernel=14, stride=14)                 │
│  • Splits image into 16×16 grid of patches                   │
│  • Each patch: 14×14 pixels                                  │
│  • Output: 256 patch embeddings                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│       ROTARY POSITION EMBEDDING (RoPE)                       │
│  • Generate sin/cos embeddings for 256 positions             │
│  • Applied in attention mechanism (Q, K)                     │
│  • Encodes 2D spatial relationships                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   TRANSFORMER BLOCK 1   │
         │  ┌─────────────────┐   │
         │  │  LayerNorm      │   │
         │  └────────┬────────┘   │
         │           ▼             │
         │  ┌─────────────────┐   │
         │  │ Multi-Head Attn │   │
         │  │  + RoPE         │   │
         │  └────────┬────────┘   │
         │           │ (residual)  │
         │  ┌────────┴────────┐   │
         │  │      Add        │   │
         │  └────────┬────────┘   │
         │           ▼             │
         │  ┌─────────────────┐   │
         │  │  LayerNorm      │   │
         │  └────────┬────────┘   │
         │           ▼             │
         │  ┌─────────────────┐   │
         │  │   MLP (FFN)     │   │
         │  └────────┬────────┘   │
         │           │ (residual)  │
         │  ┌────────┴────────┐   │
         │  │      Add        │   │
         │  └────────┬────────┘   │
         └───────────┼────────────┘
                     │
                     ▼
                    ...
                     │
                     ▼
         ┌────────────────────────┐
         │   TRANSFORMER BLOCK N   │
         │   (same structure)      │
         └───────────┬─────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              FINAL LAYER NORM                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         OPTIONAL TOKEN PROJECTION                            │
│  • None: Keep all 256 tokens                                 │
│  • Mean: Average pooling → 1 token                           │
│  • Max: Max pooling → 1 token                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              VISUAL TOKENS OUTPUT                            │
│         (batch_size × num_tokens × embed_dim)                │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Component Descriptions

### 1. Patch Embedding

**Purpose**: Convert image into a sequence of patch embeddings.

**Implementation**:
- Uses Conv2d as a learnable linear projection
- Kernel size = patch size (14×14)
- Stride = patch size (no overlap)
- Flattens spatial dimensions to create sequence

**Input/Output**:
- Input: (B, 3, 224, 224)
- Output: (B, 256, embed_dim)

**Key Points**:
- 224 / 14 = 16 patches per dimension
- Total: 16 × 16 = 256 patches
- Each patch represents a 14×14 region of the image

### 2. Rotary Position Embedding (RoPE)

**Purpose**: Encode spatial position information with better generalization.

**How it works**:
- Generates sin/cos embeddings for each position
- Applied to Query (Q) and Key (K) in attention
- Preserves relative position information
- Better than absolute position embeddings for varying sequence lengths

**Advantages over Absolute PE**:
- Encodes relative distances between tokens
- Better extrapolation to longer sequences
- No learnable parameters (deterministic)

**Mathematics**:
```
For position i and dimension d:
cos_emb[i, d] = cos(i / 10000^(2d/D))
sin_emb[i, d] = sin(i / 10000^(2d/D))

Apply rotation to Q and K:
Q_rotated = Q * cos + rotate_half(Q) * sin
K_rotated = K * cos + rotate_half(K) * sin
```

### 3. Multi-Head Self-Attention

**Purpose**: Capture relationships between different patches.

**Structure**:
- Multiple attention heads (8-16 depending on model)
- Each head has dimension = embed_dim / num_heads
- Parallel processing of different representation subspaces

**Process**:
1. Linear projections to Q, K, V
2. Apply RoPE to Q and K
3. Compute attention scores: Q @ K^T / sqrt(d)
4. Apply softmax
5. Weight values: Attention @ V
6. Concatenate heads and project

**Attention Mechanism**:
```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
```

### 4. Feed-Forward Network (MLP)

**Purpose**: Non-linear transformation of attention outputs.

**Structure**:
- Two linear layers
- GELU activation (smooth, differentiable)
- Expansion ratio: typically 4× (hidden_dim = 4 × embed_dim)

**Process**:
```
FFN(x) = Linear2(GELU(Linear1(x)))
where:
  Linear1: embed_dim → 4 × embed_dim
  Linear2: 4 × embed_dim → embed_dim
```

### 5. Layer Normalization

**Purpose**: Stabilize training and improve convergence.

**Properties**:
- Normalizes across feature dimension
- Learnable scale (γ) and shift (β) parameters
- Applied before attention and MLP (Pre-LN)

**Formula**:
```
LN(x) = γ * (x - μ) / √(σ² + ε) + β
```

### 6. Residual Connections

**Purpose**: Enable gradient flow in deep networks.

**Implementation**:
- Add input to output: `x = x + Attention(x)`
- Allows training of very deep models (24-32 layers)
- Helps preserve information from early layers

## Data Flow Example

Let's trace a single image through the pipeline:

### Input
```
Image: (224, 224, 3) - RGB image
```

### After Preprocessing
```
Tensor: (1, 3, 224, 224) - Normalized tensor with batch dimension
```

### After Patch Embedding
```
Patches: (1, 256, 1024) - 256 patch tokens, each 1024-dimensional
```

### Through Transformer Layers
```
Layer 1:  (1, 256, 1024)
  → Attention → (1, 256, 1024)
  → Add → (1, 256, 1024)
  → FFN → (1, 256, 1024)
  → Add → (1, 256, 1024)

Layer 2:  (1, 256, 1024)
  ...
  → (1, 256, 1024)

...

Layer 24: (1, 256, 1024)
  → Final output: (1, 256, 1024)
```

### Output
```
Visual Tokens: (1, 256, 1024)
  - 1: Batch size
  - 256: Number of visual tokens
  - 1024: Embedding dimension
```

## Model Configurations

### Small Model
- **Embedding Dim**: 512
- **Depth**: 12 layers
- **Attention Heads**: 8
- **Parameters**: ~38M
- **Use Case**: Fast inference, limited compute

### Base Model  
- **Embedding Dim**: 1024
- **Depth**: 24 layers
- **Attention Heads**: 16
- **Parameters**: ~303M
- **Use Case**: Balanced performance/accuracy

### Large Model
- **Embedding Dim**: 1280
- **Depth**: 32 layers
- **Attention Heads**: 16
- **Parameters**: ~630M
- **Use Case**: Maximum accuracy, high compute

## Integration with Language Model

The visual tokens from the ViT can be integrated with a language model:

```
┌──────────────┐     ┌──────────────┐
│    Image     │────▶│   ViT (This) │
└──────────────┘     └──────┬───────┘
                            │
                            ▼
                     Visual Tokens
                      (256 × 1024)
                            │
                            ▼
                  ┌─────────────────┐
                  │  Linear Proj    │
                  │  1024 → 4096    │
                  └────────┬────────┘
                           │
                           ▼
                  Projected Tokens
                    (256 × 4096)
                           │
         ┌─────────────────┴──────────────────┐
         │                                     │
         ▼                                     ▼
   Text Tokens                         Visual Tokens
   (N × 4096)                          (256 × 4096)
         │                                     │
         └─────────────────┬───────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Language Model  │
                  │   (Qwen3-LM)    │
                  └────────┬────────┘
                           │
                           ▼
                    Text Generation
```

## Performance Characteristics

### Computational Complexity

For an input image of size H×W with patch size P:
- Number of patches: N = (H/P) × (W/P)
- Attention complexity: O(N² × D) per layer
- Total: O(L × N² × D) where L = depth, D = embed_dim

For 224×224 images with 14×14 patches:
- N = 256 patches
- Attention: O(256² × D) ≈ O(65K × D) operations
- Manageable for modern GPUs

### Memory Requirements

Approximate memory for batch size 1:
- Small (512-dim, 12 layers): ~1 GB
- Base (1024-dim, 24 layers): ~2 GB  
- Large (1280-dim, 32 layers): ~4 GB

## Key Design Choices

1. **RoPE instead of Absolute PE**
   - Better generalization
   - Handles variable resolutions
   - No extra parameters

2. **Pre-Normalization**
   - More stable training
   - Better gradient flow
   - Standard in modern transformers

3. **No CLS Token**
   - Uses all patch tokens
   - More spatial information
   - Suitable for dense prediction

4. **GELU Activation**
   - Smoother than ReLU
   - Better gradients
   - Standard in transformers

5. **Patch Size 14×14**
   - Balance between:
     * Too small: Too many tokens (slow)
     * Too large: Lost spatial detail
   - 256 tokens manageable for attention

## Summary

The Qwen3-VL ViT is a carefully designed vision encoder that:
- Converts images into rich visual token representations
- Uses modern techniques (RoPE, Pre-LN, GELU)
- Scales to different model sizes
- Integrates seamlessly with language models
- Provides flexible output options (pooling strategies)

The implementation in this repository provides a complete, working version of this architecture with clear documentation and examples.
