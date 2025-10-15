"""
Configuration file for Qwen3-VL Vision Transformer models.

This file defines model configurations and hyperparameters.
"""

# Model configurations for different sizes
MODEL_CONFIGS = {
    "small": {
        "img_size": 224,
        "patch_size": 14,
        "in_channels": 3,
        "embed_dim": 512,
        "depth": 12,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "description": "Small ViT model (~22M parameters) - Fast inference",
    },
    "base": {
        "img_size": 224,
        "patch_size": 14,
        "in_channels": 3,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "description": "Base ViT model (~86M parameters) - Balanced performance",
    },
    "large": {
        "img_size": 224,
        "patch_size": 14,
        "in_channels": 3,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "description": "Large ViT model (~304M parameters) - Best accuracy",
    },
}

# Image preprocessing settings
PREPROCESSING_CONFIG = {
    "mean": (0.485, 0.456, 0.406),  # ImageNet normalization mean
    "std": (0.229, 0.224, 0.225),   # ImageNet normalization std
    "interpolation": "bicubic",      # Resize interpolation method
}

# Training hyperparameters (for reference)
TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
    "batch_size": 256,
    "epochs": 300,
    "warmup_epochs": 20,
    "optimizer": "adamw",
    "scheduler": "cosine",
}

# Inference settings
INFERENCE_CONFIG = {
    "batch_size": 8,
    "num_workers": 4,
    "pin_memory": True,
    "use_amp": True,  # Automatic Mixed Precision
}

# Qwen3-VL specific settings
QWEN3_VL_CONFIG = {
    "vision_model": {
        "type": "vit",
        "img_size": 224,
        "patch_size": 14,
        "num_patches": 256,  # (224/14)^2
        "embed_dim": 1024,
        "position_encoding": "rope",  # Rotary Position Embedding
    },
    "integration": {
        "visual_token_dim": 1024,
        "language_model_dim": 4096,  # Typical for Qwen3
        "projection_type": "linear",
        "use_all_tokens": True,  # Use all 256 patch tokens
    },
}


def get_model_config(model_size: str) -> dict:
    """
    Get configuration for a specific model size.
    
    Args:
        model_size: One of "small", "base", "large"
    
    Returns:
        Configuration dictionary
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_size].copy()


def get_preprocessing_config() -> dict:
    """Get preprocessing configuration."""
    return PREPROCESSING_CONFIG.copy()


def get_qwen3_vl_config() -> dict:
    """Get Qwen3-VL specific configuration."""
    return QWEN3_VL_CONFIG.copy()


def print_config(config_name: str = "all"):
    """
    Print configuration details.
    
    Args:
        config_name: Which config to print ("all", "models", "preprocessing", "qwen3vl")
    """
    if config_name in ["all", "models"]:
        print("\n" + "="*80)
        print("MODEL CONFIGURATIONS")
        print("="*80)
        for size, config in MODEL_CONFIGS.items():
            print(f"\n{size.upper()}:")
            for key, value in config.items():
                print(f"  {key}: {value}")
    
    if config_name in ["all", "preprocessing"]:
        print("\n" + "="*80)
        print("PREPROCESSING CONFIGURATION")
        print("="*80)
        for key, value in PREPROCESSING_CONFIG.items():
            print(f"  {key}: {value}")
    
    if config_name in ["all", "qwen3vl"]:
        print("\n" + "="*80)
        print("QWEN3-VL CONFIGURATION")
        print("="*80)
        for section, settings in QWEN3_VL_CONFIG.items():
            print(f"\n{section.upper()}:")
            for key, value in settings.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    # Print all configurations
    print_config("all")
