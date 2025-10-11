"""
Test suite for Qwen3-VL Vision Transformer implementation.

Run with: python test_implementation.py
"""

import torch
import numpy as np
from PIL import Image
import sys
import traceback


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*80)
    print("TEST 1: Module Imports")
    print("="*80)
    
    try:
        from qwen3_vl_vit import (
            PatchEmbed,
            RotaryPositionEmbedding,
            MultiHeadAttention,
            MLP,
            TransformerBlock,
            Qwen3VisionTransformer,
            create_qwen3_vl_vit,
        )
        print("✓ qwen3_vl_vit module imported successfully")
        
        from image_to_token_pipeline import (
            ImagePreprocessor,
            TokenPostprocessor,
            Qwen3VLPipeline,
        )
        print("✓ image_to_token_pipeline module imported successfully")
        
        from config import (
            MODEL_CONFIGS,
            get_model_config,
            get_preprocessing_config,
        )
        print("✓ config module imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_patch_embed():
    """Test patch embedding module."""
    print("\n" + "="*80)
    print("TEST 2: Patch Embedding")
    print("="*80)
    
    try:
        from qwen3_vl_vit import PatchEmbed
        
        patch_embed = PatchEmbed(img_size=224, patch_size=14, in_channels=3, embed_dim=512)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        out = patch_embed(x)
        
        expected_shape = (2, 256, 512)  # (batch, num_patches, embed_dim)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {out.shape}")
        print(f"✓ Number of patches: {patch_embed.num_patches}")
        return True
    except Exception as e:
        print(f"✗ Patch embedding test failed: {e}")
        traceback.print_exc()
        return False


def test_rope():
    """Test Rotary Position Embedding."""
    print("\n" + "="*80)
    print("TEST 3: Rotary Position Embedding")
    print("="*80)
    
    try:
        from qwen3_vl_vit import RotaryPositionEmbedding
        
        rope = RotaryPositionEmbedding(dim=64)
        
        # Generate embeddings
        cos, sin = rope(256)  # 256 positions
        
        assert cos.shape == (256, 64), f"Expected (256, 64), got {cos.shape}"
        assert sin.shape == (256, 64), f"Expected (256, 64), got {sin.shape}"
        
        print(f"✓ Cosine embeddings shape: {cos.shape}")
        print(f"✓ Sine embeddings shape: {sin.shape}")
        return True
    except Exception as e:
        print(f"✗ RoPE test failed: {e}")
        traceback.print_exc()
        return False


def test_attention():
    """Test Multi-Head Attention."""
    print("\n" + "="*80)
    print("TEST 4: Multi-Head Attention")
    print("="*80)
    
    try:
        from qwen3_vl_vit import MultiHeadAttention
        
        attn = MultiHeadAttention(embed_dim=512, num_heads=8)
        
        # Test forward pass
        x = torch.randn(2, 256, 512)  # (batch, seq_len, embed_dim)
        out = attn(x)
        
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {out.shape}")
        print(f"✓ Number of heads: {attn.num_heads}")
        return True
    except Exception as e:
        print(f"✗ Attention test failed: {e}")
        traceback.print_exc()
        return False


def test_transformer_block():
    """Test Transformer Block."""
    print("\n" + "="*80)
    print("TEST 5: Transformer Block")
    print("="*80)
    
    try:
        from qwen3_vl_vit import TransformerBlock
        
        block = TransformerBlock(embed_dim=512, num_heads=8, mlp_ratio=4.0)
        
        # Test forward pass
        x = torch.randn(2, 256, 512)
        out = block(x)
        
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {out.shape}")
        return True
    except Exception as e:
        print(f"✗ Transformer block test failed: {e}")
        traceback.print_exc()
        return False


def test_vision_transformer():
    """Test complete Vision Transformer."""
    print("\n" + "="*80)
    print("TEST 6: Vision Transformer")
    print("="*80)
    
    try:
        from qwen3_vl_vit import Qwen3VisionTransformer
        
        model = Qwen3VisionTransformer(
            img_size=224,
            patch_size=14,
            in_channels=3,
            embed_dim=512,
            depth=6,  # Smaller depth for testing
            num_heads=8,
        )
        model.eval()
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        
        expected_shape = (2, 256, 512)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {out.shape}")
        print(f"✓ Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Vision Transformer test failed: {e}")
        traceback.print_exc()
        return False


def test_model_factory():
    """Test model factory function."""
    print("\n" + "="*80)
    print("TEST 7: Model Factory")
    print("="*80)
    
    try:
        from qwen3_vl_vit import create_qwen3_vl_vit
        
        for size in ["small", "base", "large"]:
            model = create_qwen3_vl_vit(model_size=size)
            model.eval()
            
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
            
            print(f"✓ {size.upper()} model:")
            print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  - Output shape: {out.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model factory test failed: {e}")
        traceback.print_exc()
        return False


def test_image_preprocessor():
    """Test image preprocessing."""
    print("\n" + "="*80)
    print("TEST 8: Image Preprocessor")
    print("="*80)
    
    try:
        from image_to_token_pipeline import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(img_size=224)
        
        # Test with numpy array
        image_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = preprocessor.preprocess(image_np)
        
        expected_shape = (1, 3, 224, 224)
        assert tensor.shape == expected_shape, f"Expected {expected_shape}, got {tensor.shape}"
        
        print(f"✓ Input: numpy array {image_np.shape}")
        print(f"✓ Output: tensor {tensor.shape}")
        
        # Test batch preprocessing
        images = [image_np, image_np, image_np]
        batch = preprocessor.preprocess_batch(images)
        
        expected_batch_shape = (3, 3, 224, 224)
        assert batch.shape == expected_batch_shape, f"Expected {expected_batch_shape}, got {batch.shape}"
        
        print(f"✓ Batch output: {batch.shape}")
        return True
    except Exception as e:
        print(f"✗ Image preprocessor test failed: {e}")
        traceback.print_exc()
        return False


def test_token_postprocessor():
    """Test token post-processing."""
    print("\n" + "="*80)
    print("TEST 9: Token Postprocessor")
    print("="*80)
    
    try:
        from image_to_token_pipeline import TokenPostprocessor
        
        tokens = torch.randn(2, 256, 512)
        
        # Test different pooling strategies
        strategies = ["none", "mean", "max"]
        for strategy in strategies:
            postprocessor = TokenPostprocessor(pooling_strategy=strategy)
            out = postprocessor.process(tokens)
            
            if strategy == "none":
                expected_shape = (2, 256, 512)
            else:
                expected_shape = (2, 1, 512)
            
            assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
            print(f"✓ Pooling '{strategy}': {out.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Token postprocessor test failed: {e}")
        traceback.print_exc()
        return False


def test_pipeline():
    """Test complete pipeline."""
    print("\n" + "="*80)
    print("TEST 10: Complete Pipeline")
    print("="*80)
    
    try:
        from image_to_token_pipeline import Qwen3VLPipeline
        
        pipeline = Qwen3VLPipeline(model_size="small")
        
        # Test single image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tokens = pipeline.process_image(image)
        
        print(f"✓ Single image processing:")
        print(f"  - Input: {image.shape}")
        print(f"  - Output: {tokens.shape}")
        
        # Test batch processing
        images = [image, image, image]
        batch_tokens = pipeline.process_batch(images)
        
        print(f"✓ Batch processing:")
        print(f"  - Input: {len(images)} images")
        print(f"  - Output: {batch_tokens.shape}")
        
        # Test numpy output
        tokens_np = pipeline.process_image(image, return_numpy=True)
        assert isinstance(tokens_np, np.ndarray), "Expected numpy array"
        
        print(f"✓ NumPy output: {type(tokens_np)}")
        
        # Test pipeline info
        info = pipeline.get_pipeline_info()
        print(f"✓ Pipeline info retrieved: {len(info)} keys")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        traceback.print_exc()
        return False


def test_config():
    """Test configuration module."""
    print("\n" + "="*80)
    print("TEST 11: Configuration")
    print("="*80)
    
    try:
        from config import get_model_config, MODEL_CONFIGS
        
        for size in ["small", "base", "large"]:
            config = get_model_config(size)
            print(f"✓ {size.upper()} config loaded:")
            print(f"  - Embed dim: {config['embed_dim']}")
            print(f"  - Depth: {config['depth']}")
            print(f"  - Num heads: {config['num_heads']}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("QWEN3-VL VISION TRANSFORMER TEST SUITE")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Patch Embedding", test_patch_embed),
        ("Rotary Position Embedding", test_rope),
        ("Multi-Head Attention", test_attention),
        ("Transformer Block", test_transformer_block),
        ("Vision Transformer", test_vision_transformer),
        ("Model Factory", test_model_factory),
        ("Image Preprocessor", test_image_preprocessor),
        ("Token Postprocessor", test_token_postprocessor),
        ("Complete Pipeline", test_pipeline),
        ("Configuration", test_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    print("="*80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
