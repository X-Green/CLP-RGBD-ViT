"""
Image to Token Pipeline for Qwen3-VL

This module provides a complete pipeline for processing images and extracting
visual tokens using the Qwen3-VL Vision Transformer.

Pipeline stages:
1. Image Preprocessing: Resize, normalize, and prepare image
2. Vision Transformer: Extract visual features
3. Token Generation: Convert visual features to tokens

Usage:
    from image_to_token_pipeline import Qwen3VLPipeline
    
    pipeline = Qwen3VLPipeline(model_size="base")
    tokens = pipeline.process_image("path/to/image.jpg")
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Union, Optional, List
import os

from qwen3_vl_vit import Qwen3VisionTransformer, create_qwen3_vl_vit


class ImagePreprocessor:
    """
    Image preprocessing for Qwen3-VL.
    
    Handles image loading, resizing, normalization following ImageNet stats.
    """
    def __init__(
        self,
        img_size: int = 224,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        self.img_size = img_size
        self.mean = mean
        self.std = std
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from file path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        return image
    
    def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for ViT input.
        
        Args:
            image: Can be:
                - str: Path to image file
                - PIL.Image: PIL Image object
                - np.ndarray: Numpy array in HWC format
        
        Returns:
            Preprocessed image tensor of shape (1, C, H, W)
        """
        # Convert to PIL Image if needed
        if isinstance(image, str):
            image = self.load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def preprocess_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images
        
        Returns:
            Batch of preprocessed images of shape (B, C, H, W)
        """
        tensors = [self.preprocess(img).squeeze(0) for img in images]
        return torch.stack(tensors, dim=0)


class TokenPostprocessor:
    """
    Post-processing for visual tokens.
    
    Handles token selection, pooling, and formatting for downstream models.
    """
    def __init__(self, pooling_strategy: str = "none"):
        """
        Args:
            pooling_strategy: One of:
                - "none": Use all tokens
                - "mean": Average pooling across spatial dimension
                - "max": Max pooling across spatial dimension
                - "cls": Use first token (if available)
        """
        self.pooling_strategy = pooling_strategy
    
    def process(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process tokens according to pooling strategy.
        
        Args:
            tokens: Token tensor of shape (B, N, D)
        
        Returns:
            Processed tokens
        """
        if self.pooling_strategy == "none":
            return tokens
        elif self.pooling_strategy == "mean":
            return tokens.mean(dim=1, keepdim=True)
        elif self.pooling_strategy == "max":
            return tokens.max(dim=1, keepdim=True)[0]
        elif self.pooling_strategy == "cls":
            # Return first token
            return tokens[:, :1, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def to_numpy(self, tokens: torch.Tensor) -> np.ndarray:
        """Convert tokens to numpy array."""
        return tokens.detach().cpu().numpy()
    
    def to_list(self, tokens: torch.Tensor) -> List[List[float]]:
        """Convert tokens to nested list."""
        return tokens.detach().cpu().tolist()


class Qwen3VLPipeline:
    """
    Complete pipeline for processing images to tokens using Qwen3-VL ViT.
    
    This pipeline encapsulates:
    1. Image preprocessing
    2. Vision Transformer inference
    3. Token post-processing
    
    Example:
        >>> pipeline = Qwen3VLPipeline(model_size="base")
        >>> tokens = pipeline.process_image("image.jpg")
        >>> print(tokens.shape)  # (1, num_patches, embed_dim)
    """
    def __init__(
        self,
        model_size: str = "base",
        img_size: int = 224,
        patch_size: int = 14,
        pooling_strategy: str = "none",
        device: Optional[str] = None,
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            model_size: ViT model size ("small", "base", "large")
            img_size: Input image size
            patch_size: Patch size for ViT
            pooling_strategy: Token pooling strategy
            device: Device to run inference on (cuda/cpu)
            pretrained_path: Path to pretrained weights (optional)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(img_size=img_size)
        
        # Initialize model
        self.model = create_qwen3_vl_vit(
            model_size=model_size,
            img_size=img_size,
            patch_size=patch_size,
        )
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize postprocessor
        self.postprocessor = TokenPostprocessor(pooling_strategy=pooling_strategy)
        
        print(f"Qwen3-VL Pipeline initialized:")
        print(f"  Model size: {model_size}")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Number of patches: {self.model.get_num_patches()}")
        print(f"  Embedding dimension: {self.model.get_embed_dim()}")
        print(f"  Device: {self.device}")
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained model weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    def process_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Process a single image and extract visual tokens.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            return_numpy: If True, return numpy array instead of tensor
        
        Returns:
            Visual tokens of shape (1, N, D) where:
                N = number of tokens (depends on pooling strategy)
                D = embedding dimension
        """
        # Preprocess image
        image_tensor = self.preprocessor.preprocess(image)
        image_tensor = image_tensor.to(self.device)
        
        # Extract tokens using ViT
        with torch.no_grad():
            tokens = self.model(image_tensor)
        
        # Post-process tokens
        tokens = self.postprocessor.process(tokens)
        
        # Convert to numpy if requested
        if return_numpy:
            return self.postprocessor.to_numpy(tokens)
        
        return tokens
    
    def process_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Process a batch of images and extract visual tokens.
        
        Args:
            images: List of images
            return_numpy: If True, return numpy array instead of tensor
        
        Returns:
            Visual tokens of shape (B, N, D)
        """
        # Preprocess batch
        image_batch = self.preprocessor.preprocess_batch(images)
        image_batch = image_batch.to(self.device)
        
        # Extract tokens using ViT
        with torch.no_grad():
            tokens = self.model(image_batch)
        
        # Post-process tokens
        tokens = self.postprocessor.process(tokens)
        
        # Convert to numpy if requested
        if return_numpy:
            return self.postprocessor.to_numpy(tokens)
        
        return tokens
    
    def get_pipeline_info(self) -> dict:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary with pipeline details
        """
        return {
            "model_info": {
                "num_patches": self.model.get_num_patches(),
                "embed_dim": self.model.get_embed_dim(),
                "img_size": self.model.img_size,
                "patch_size": self.model.patch_size,
            },
            "device": str(self.device),
            "pooling_strategy": self.postprocessor.pooling_strategy,
        }


def visualize_tokens(
    tokens: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Visual Tokens"
):
    """
    Visualize token embeddings (requires matplotlib).
    
    Args:
        tokens: Token tensor of shape (B, N, D) or (N, D)
        save_path: Path to save visualization (optional)
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib is required for visualization. Install with: pip install matplotlib")
        return
    
    # Convert to numpy and remove batch dimension if present
    if tokens.dim() == 3:
        tokens = tokens[0]
    tokens_np = tokens.detach().cpu().numpy()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot token norms
    token_norms = np.linalg.norm(tokens_np, axis=1)
    ax1.bar(range(len(token_norms)), token_norms)
    ax1.set_xlabel("Token Index")
    ax1.set_ylabel("L2 Norm")
    ax1.set_title("Token Magnitudes")
    
    # Plot token embeddings heatmap
    im = ax2.imshow(tokens_np.T, aspect='auto', cmap='viridis')
    ax2.set_xlabel("Token Index")
    ax2.set_ylabel("Embedding Dimension")
    ax2.set_title("Token Embeddings")
    plt.colorbar(im, ax=ax2)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Qwen3-VL Image-to-Token Pipeline Example")
    print("=" * 80)
    
    # Create pipeline
    pipeline = Qwen3VLPipeline(
        model_size="base",
        img_size=224,
        patch_size=14,
        pooling_strategy="none",
    )
    
    # Get pipeline info
    info = pipeline.get_pipeline_info()
    print("\nPipeline Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create a dummy image for demonstration
    print("\nCreating dummy image for demonstration...")
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Process image
    print("\nProcessing image...")
    tokens = pipeline.process_image(dummy_image)
    
    print(f"\nOutput tokens shape: {tokens.shape}")
    print(f"Output tokens dtype: {tokens.dtype}")
    print(f"Output tokens device: {tokens.device}")
    
    # Display statistics
    print("\nToken Statistics:")
    print(f"  Mean: {tokens.mean().item():.6f}")
    print(f"  Std: {tokens.std().item():.6f}")
    print(f"  Min: {tokens.min().item():.6f}")
    print(f"  Max: {tokens.max().item():.6f}")
    
    print("\n" + "=" * 80)
    print("Pipeline example completed successfully!")
    print("=" * 80)
