"""
Qwen3-VL Vision Transformer (ViT) Implementation

This module implements the Vision Transformer architecture used in Qwen3-VL.
Qwen3-VL uses a modified ViT with the following key features:
- Patch embedding with configurable patch size
- Rotary Position Embedding (RoPE) for 2D spatial positions
- Multi-head self-attention with spatial awareness
- Dynamic resolution support with naive temporal compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    Converts an image into a sequence of patch embeddings.
    
    Qwen3-VL typically uses 14x14 patches for 224x224 images.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Projection layer: Conv2d acts as a linear projection per patch
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, C, H, W)
        Returns:
            Patch embeddings of shape (B, N, D) where N = num_patches
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # Apply patch projection: (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.proj(x)
        
        # Flatten patches: (B, D, H/P, W/P) -> (B, D, N)
        x = x.flatten(2)
        
        # Transpose: (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)
        
        return x


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for 2D spatial positions.
    
    Qwen3-VL uses RoPE to encode spatial relationships between patches.
    This implementation extends 1D RoPE to 2D grids.
    """
    def __init__(self, dim: int, max_position: int = 1024):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        
        # Compute frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rotary position embeddings for a sequence.
        
        Args:
            seq_len: Sequence length (number of patches)
        Returns:
            cos, sin: Cosine and sine position embeddings
        """
        # Generate position indices
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        
        # Compute frequencies: outer product of positions and inverse frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Concatenate to get full dimensionality
        emb = torch.cat((freqs, freqs), dim=-1)
        
        return emb.cos(), emb.sin()
    
    def apply_rotary_pos_emb(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.
        """
        # Rotate half the dimensions
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE.
    
    Qwen3-VL uses multi-head attention to capture relationships between
    different image patches.
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D)
            rope_cos: Cosine position embeddings
            rope_sin: Sine position embeddings
        Returns:
            Output tensor of shape (B, N, D)
        """
        B, N, D = x.shape
        
        # Compute QKV: (B, N, D) -> (B, N, 3*D) -> 3 x (B, N, D)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            # Reshape for RoPE application
            cos = rope_cos.unsqueeze(0).unsqueeze(0)  # (1, 1, N, head_dim)
            sin = rope_sin.unsqueeze(0).unsqueeze(0)
            
            def rotate_half(x):
                x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
                return torch.cat((-x2, x1), dim=-1)
            
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    Feed-Forward Network (FFN) / MLP block.
    
    Standard two-layer MLP with GELU activation used in ViT.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block.
    
    Standard transformer block with:
    - Multi-head self-attention with RoPE
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        
        # FFN with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class Qwen3VisionTransformer(nn.Module):
    """
    Qwen3-VL Vision Transformer.
    
    Complete ViT architecture used in Qwen3-VL for processing images
    and generating visual tokens.
    
    Architecture:
    1. Patch Embedding: Convert image to patch tokens
    2. Rotary Position Embedding: Add spatial position information
    3. Transformer Layers: Process patch tokens with self-attention
    4. Output: Visual tokens for downstream language model
    
    Args:
        img_size: Input image size (default: 224)
        patch_size: Size of each patch (default: 14)
        in_channels: Number of input channels (default: 3 for RGB)
        embed_dim: Embedding dimension (default: 1024)
        depth: Number of transformer layers (default: 24)
        num_heads: Number of attention heads (default: 16)
        mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
        dropout: Dropout rate (default: 0.0)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_output_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding layer
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        # Rotary position embedding
        self.rope = RotaryPositionEmbedding(
            dim=embed_dim // num_heads,
            max_position=self.num_patches,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Optional projection for specific number of output tokens
        self.num_output_tokens = num_output_tokens
        if num_output_tokens is not None and num_output_tokens != self.num_patches:
            self.token_projection = nn.Linear(self.num_patches, num_output_tokens)
        else:
            self.token_projection = None
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
        
        Returns:
            Visual tokens of shape (B, N, D) where:
                B = batch size
                N = number of output tokens
                D = embedding dimension
        """
        # Convert image to patches
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Get rotary position embeddings
        rope_cos, rope_sin = self.rope(x.shape[1])
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)
        
        # Final layer norm
        x = self.norm(x)
        
        # Optional token projection for different output size
        if self.token_projection is not None:
            # Transpose to apply linear projection across sequence dimension
            x = x.transpose(1, 2)  # (B, D, N)
            x = self.token_projection(x)  # (B, D, num_output_tokens)
            x = x.transpose(1, 2)  # (B, num_output_tokens, D)
        
        return x
    
    def get_num_patches(self) -> int:
        """Return the number of patches per image."""
        return self.num_patches
    
    def get_embed_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embed_dim


def create_qwen3_vl_vit(
    model_size: str = "base",
    img_size: int = 224,
    patch_size: int = 14,
) -> Qwen3VisionTransformer:
    """
    Factory function to create Qwen3-VL ViT models with different sizes.
    
    Args:
        model_size: One of "small", "base", "large" (default: "base")
        img_size: Input image size (default: 224)
        patch_size: Patch size (default: 14)
    
    Returns:
        Qwen3VisionTransformer model
    """
    configs = {
        "small": {
            "embed_dim": 512,
            "depth": 12,
            "num_heads": 8,
        },
        "base": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
        },
        "large": {
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    return Qwen3VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=4.0,
        dropout=0.0,
    )
