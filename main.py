#!/usr/bin/env python3
"""
Z-Image-Turbo inference script using diffusers library
Uses the original 32GB model with full quality
"""

import sys
from pathlib import Path

import torch
from diffusers import ZImagePipeline


def main():
    # Configuration
    model_id = "Tongyi-MAI/Z-Image-Turbo"  # HuggingFace model ID
    prompt = input("prompt > ")

    negative_prompt = ""
    output_path = "output.png"

    # Generation parameters
    width = 1024
    height = 1024
    num_inference_steps = 9  # Z-Image-Turbo recommended steps
    guidance_scale = 0.0
    seed = 43  # Set to a number for reproducible results

    print("=" * 70)
    print("Z-Image-Turbo Image Generation")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}")
    print(f"Steps: {num_inference_steps}")
    print(f"Guidance Scale: {guidance_scale}")
    print("=" * 70)

    # Check if MPS (Metal Performance Shaders) is available on Mac
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
        print(f"\n✓ Using Metal Performance Shaders (MPS) on Apple Silicon")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"\n✓ Using CUDA GPU")
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"\n⚠️  Using CPU (will be slower)")

    try:
        # Load the pipeline
        print(f"\n📦 Loading model (first time will download ~32GB)...")
        print("   This may take several minutes...")

        pipeline = ZImagePipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=False,
        )

        # Move to device
        pipeline = pipeline.to(device)

        # Enable memory optimizations
        if device == "mps" or device == "cuda":
            try:
                pipeline.enable_attention_slicing()
                print("✓ Memory optimization enabled")
            except:
                pass

        print("✓ Model loaded successfully!")

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            print(f"✓ Using seed: {seed}")

        # Generate image
        print(f"\n🎨 Generating image...")

        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # Save the image
        image.save(output_path)
        print(f"\n✅ Image saved to: {output_path}")
        print(f"   Resolution: {image.size[0]}x{image.size[1]}")

    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have enough disk space (~32GB for model)")
        print("2. Check your internet connection for first-time download")
        print("3. Try reducing resolution if you run out of memory")
        print("4. Make sure diffusers is installed: uv sync")
        sys.exit(1)


if __name__ == "__main__":
    main()
