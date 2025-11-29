#!/usr/bin/env python3
"""
Debug script to test Z-Image-Turbo with different configurations
Helps identify the best device/dtype combination
"""

import torch
from diffusers import ZImagePipeline


def test_generation(device, dtype, dtype_name):
    """Test image generation with specific device/dtype"""
    print(f"\n{'='*70}")
    print(f"Testing: {device.upper()} with {dtype_name}")
    print(f"{'='*70}")

    try:
        print("Loading model...")
        pipeline = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            use_safetensors=True,
            trust_remote_code=True,
        )

        pipeline = pipeline.to(device)

        print("Model loaded successfully")

        # Simple test prompt
        prompt = "a red apple on a wooden table"

        print(f"Generating test image with prompt: '{prompt}'")

        generator = torch.Generator(device=device).manual_seed(42)

        image = pipeline(
            prompt=prompt,
            height=512,  # Smaller for faster testing
            width=512,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

        # Check if image is all black
        import numpy as np
        img_array = np.array(image)
        mean_value = img_array.mean()
        max_value = img_array.max()
        min_value = img_array.min()

        print(f"Image stats - Mean: {mean_value:.2f}, Min: {min_value}, Max: {max_value}")

        if max_value < 10:
            print("❌ RESULT: Image is all black (FAILED)")
            return False
        else:
            output_path = f"test_{device}_{dtype_name}.png"
            image.save(output_path)
            print(f"✅ RESULT: Image generated successfully (PASSED)")
            print(f"   Saved to: {output_path}")
            return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    print("Z-Image-Turbo Configuration Tester")
    print("This will test different device/dtype combinations\n")

    results = {}

    # Test configurations
    if torch.backends.mps.is_available():
        print("\n🍎 MPS (Apple Silicon) detected")

        # MPS with float32 (most compatible)
        results["MPS + float32"] = test_generation("mps", torch.float32, "float32")

        # MPS with float16
        results["MPS + float16"] = test_generation("mps", torch.float16, "float16")

    if torch.cuda.is_available():
        print("\n🎮 CUDA GPU detected")

        # CUDA with bfloat16
        results["CUDA + bfloat16"] = test_generation("cuda", torch.bfloat16, "bfloat16")

        # CUDA with float16
        results["CUDA + float16"] = test_generation("cuda", torch.float16, "float16")

    # Always test CPU as fallback
    print("\n💻 Testing CPU (fallback)")
    results["CPU + float32"] = test_generation("cpu", torch.float32, "float32")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)

    working_configs = []
    for config, passed in results.items():
        status = "✅ WORKING" if passed else "❌ FAILED"
        print(f"{config:20} : {status}")
        if passed:
            working_configs.append(config)

    print("\n" + "="*70)
    if working_configs:
        print(f"✅ Found {len(working_configs)} working configuration(s):")
        for config in working_configs:
            print(f"   - {config}")
        print("\nRECOMMENDATION:")
        print(f"Use configuration: {working_configs[0]}")
    else:
        print("❌ No working configurations found!")
        print("\nTroubleshooting suggestions:")
        print("1. Verify model files downloaded correctly")
        print("2. Check available disk space")
        print("3. Try updating diffusers: pip install --upgrade git+https://github.com/huggingface/diffusers")
        print("4. Check if model is compatible with your hardware")
    print("="*70)


if __name__ == "__main__":
    main()
