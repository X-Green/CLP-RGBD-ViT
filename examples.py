"""
Example Usage of Qwen3-VL Vision Transformer

This script demonstrates how to use the Qwen3-VL ViT implementation
for processing images and extracting visual tokens.
"""

import torch
import numpy as np
from PIL import Image

from qwen3_vl_vit import create_qwen3_vl_vit, Qwen3VisionTransformer
from image_to_token_pipeline import Qwen3VLPipeline


def example_1_basic_usage():
    """Example 1: Basic usage with dummy image"""
    print("\n" + "="*80)
    print("Example 1: Basic Usage")
    print("="*80)
    
    # Create a pipeline
    pipeline = Qwen3VLPipeline(model_size="base")
    
    # Create a dummy RGB image (224x224)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Process the image
    tokens = pipeline.process_image(dummy_image)
    
    print(f"\nInput: {dummy_image.shape} image")
    print(f"Output: {tokens.shape} tokens")
    print(f"  - Batch size: {tokens.shape[0]}")
    print(f"  - Number of tokens: {tokens.shape[1]}")
    print(f"  - Embedding dimension: {tokens.shape[2]}")


def example_2_batch_processing():
    """Example 2: Batch processing of multiple images"""
    print("\n" + "="*80)
    print("Example 2: Batch Processing")
    print("="*80)
    
    # Create a pipeline
    pipeline = Qwen3VLPipeline(model_size="small")
    
    # Create multiple dummy images
    batch_size = 4
    images = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]
    
    # Process batch
    tokens = pipeline.process_batch(images)
    
    print(f"\nProcessed {batch_size} images")
    print(f"Output shape: {tokens.shape}")


def example_3_different_pooling():
    """Example 3: Different pooling strategies"""
    print("\n" + "="*80)
    print("Example 3: Different Pooling Strategies")
    print("="*80)
    
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    pooling_strategies = ["none", "mean", "max"]
    
    for strategy in pooling_strategies:
        pipeline = Qwen3VLPipeline(
            model_size="small",
            pooling_strategy=strategy
        )
        tokens = pipeline.process_image(dummy_image)
        print(f"\nPooling: {strategy:8s} -> Output shape: {tokens.shape}")


def example_4_model_details():
    """Example 4: Inspect model architecture details"""
    print("\n" + "="*80)
    print("Example 4: Model Architecture Details")
    print("="*80)
    
    for model_size in ["small", "base", "large"]:
        model = create_qwen3_vl_vit(model_size=model_size)
        
        print(f"\n{model_size.upper()} Model:")
        print(f"  Number of patches: {model.get_num_patches()}")
        print(f"  Embedding dimension: {model.get_embed_dim()}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")


def example_5_step_by_step():
    """Example 5: Step-by-step processing"""
    print("\n" + "="*80)
    print("Example 5: Step-by-Step Processing")
    print("="*80)
    
    # Step 1: Create model
    print("\nStep 1: Creating Vision Transformer...")
    model = create_qwen3_vl_vit(model_size="base", img_size=224, patch_size=14)
    model.eval()
    print(f"  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 2: Prepare image
    print("\nStep 2: Preparing input image...")
    from image_to_token_pipeline import ImagePreprocessor
    preprocessor = ImagePreprocessor(img_size=224)
    
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image_tensor = preprocessor.preprocess(dummy_image)
    print(f"  Preprocessed image shape: {image_tensor.shape}")
    
    # Step 3: Extract visual tokens
    print("\nStep 3: Extracting visual tokens...")
    with torch.no_grad():
        tokens = model(image_tensor)
    print(f"  Output tokens shape: {tokens.shape}")
    
    # Step 4: Analyze tokens
    print("\nStep 4: Token analysis...")
    print(f"  Token mean: {tokens.mean().item():.6f}")
    print(f"  Token std: {tokens.std().item():.6f}")
    print(f"  Token range: [{tokens.min().item():.6f}, {tokens.max().item():.6f}]")


def example_6_pipeline_info():
    """Example 6: Get pipeline information"""
    print("\n" + "="*80)
    print("Example 6: Pipeline Information")
    print("="*80)
    
    pipeline = Qwen3VLPipeline(model_size="base")
    info = pipeline.get_pipeline_info()
    
    print("\nPipeline Configuration:")
    import json
    print(json.dumps(info, indent=2))


def example_7_numpy_output():
    """Example 7: Get numpy output instead of torch tensor"""
    print("\n" + "="*80)
    print("Example 7: NumPy Output")
    print("="*80)
    
    pipeline = Qwen3VLPipeline(model_size="small")
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Get numpy output
    tokens_np = pipeline.process_image(dummy_image, return_numpy=True)
    
    print(f"\nOutput type: {type(tokens_np)}")
    print(f"Output shape: {tokens_np.shape}")
    print(f"Output dtype: {tokens_np.dtype}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("QWEN3-VL VISION TRANSFORMER EXAMPLES")
    print("="*80)
    
    examples = [
        example_1_basic_usage,
        example_2_batch_processing,
        example_3_different_pooling,
        example_4_model_details,
        example_5_step_by_step,
        example_6_pipeline_info,
        example_7_numpy_output,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
