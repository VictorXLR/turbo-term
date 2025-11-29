#!/usr/bin/env python3
"""
Z-Image-Turbo Optimization Benchmark
Tests different optimization techniques for M4 Max / Apple Silicon
"""

import time
from typing import Optional

import torch
from diffusers import ZImagePipeline


class OptimizedPipeline:
    """Wrapper for optimized pipeline with various techniques"""

    def __init__(self, optimization_level: str = "standard"):
        """
        Args:
            optimization_level: "standard", "compile", "quantize", or "aggressive"
        """
        self.optimization_level = optimization_level
        self.pipeline = None
        self.device = None
        self.dtype = None

    def load(self):
        """Load and optimize the pipeline"""
        print(f"\n{'='*70}")
        print(f"Loading with optimization: {self.optimization_level.upper()}")
        print(f"{'='*70}")

        # Device setup
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.bfloat16
            print("Device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
            print(f"Device: CUDA - {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            print("Device: CPU")

        # Load base pipeline
        print("Loading base model...")
        start = time.time()

        self.pipeline = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=self.dtype,
            use_safetensors=True,
        )

        self.pipeline = self.pipeline.to(self.device)

        # Apply optimizations based on level
        if self.optimization_level == "standard":
            self._apply_standard_optimizations()
        elif self.optimization_level == "compile":
            self._apply_compile_optimizations()
        elif self.optimization_level == "quantize":
            self._apply_quantize_optimizations()
        elif self.optimization_level == "aggressive":
            self._apply_aggressive_optimizations()

        elapsed = time.time() - start
        print(f"✓ Pipeline loaded and optimized in {elapsed:.2f}s")

    def _apply_standard_optimizations(self):
        """Standard optimizations (baseline)"""
        print("Applying: Attention slicing")
        try:
            self.pipeline.enable_attention_slicing()
        except:
            pass

    def _apply_compile_optimizations(self):
        """Use torch.compile for JIT optimization"""
        print("Applying: Attention slicing + torch.compile")
        try:
            self.pipeline.enable_attention_slicing()
        except:
            pass

        print("Compiling UNet (this takes time but speeds up inference)...")
        try:
            # Compile the UNet for faster inference
            self.pipeline.unet = torch.compile(
                self.pipeline.unet,
                mode="reduce-overhead",  # Best for repeated calls
                fullgraph=False,  # More compatible
            )
            print("✓ UNet compiled successfully")
        except Exception as e:
            print(f"⚠ Compilation failed: {e}")

    def _apply_quantize_optimizations(self):
        """Apply quantization with torchao"""
        print("Applying: Attention slicing + int8 quantization")
        try:
            self.pipeline.enable_attention_slicing()
        except:
            pass

        try:
            from torchao.quantization import quantize_, int8_weight_only

            print("Quantizing model to int8...")
            # Quantize the transformer/unet
            if hasattr(self.pipeline, "transformer"):
                quantize_(self.pipeline.transformer, int8_weight_only())
                print("✓ Transformer quantized")
            elif hasattr(self.pipeline, "unet"):
                quantize_(self.pipeline.unet, int8_weight_only())
                print("✓ UNet quantized")
        except Exception as e:
            print(f"⚠ Quantization failed: {e}")

    def _apply_aggressive_optimizations(self):
        """All optimizations combined"""
        print("Applying: Attention slicing + quantization + compile")
        try:
            self.pipeline.enable_attention_slicing()
        except:
            pass

        # Try quantization first
        try:
            from torchao.quantization import quantize_, int8_weight_only

            print("Quantizing model to int8...")
            if hasattr(self.pipeline, "transformer"):
                quantize_(self.pipeline.transformer, int8_weight_only())
                print("✓ Transformer quantized")
            elif hasattr(self.pipeline, "unet"):
                quantize_(self.pipeline.unet, int8_weight_only())
                print("✓ UNet quantized")
        except Exception as e:
            print(f"⚠ Quantization failed: {e}")

        # Then compile
        try:
            print("Compiling UNet...")
            if hasattr(self.pipeline, "transformer"):
                self.pipeline.transformer = torch.compile(
                    self.pipeline.transformer, mode="reduce-overhead"
                )
            elif hasattr(self.pipeline, "unet"):
                self.pipeline.unet = torch.compile(
                    self.pipeline.unet, mode="reduce-overhead"
                )
            print("✓ Model compiled")
        except Exception as e:
            print(f"⚠ Compilation failed: {e}")

    def generate(self, prompt: str, seed: int = 43, warmup: bool = False):
        """Generate an image and return timing"""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        start = time.time()

        image = self.pipeline(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

        elapsed = time.time() - start

        if not warmup:
            print(f"  Generation time: {elapsed:.2f}s")

        return image, elapsed


def benchmark():
    """Run benchmark comparing different optimization levels"""
    print("\n" + "="*70)
    print("Z-Image-Turbo Optimization Benchmark")
    print("Testing different optimization strategies on your hardware")
    print("="*70)

    prompt = "a serene mountain landscape at sunset, photorealistic, 8k"
    seed = 43

    # Test configurations
    configs = [
        ("standard", "Standard (baseline)"),
        ("compile", "Torch Compile"),
        ("quantize", "Int8 Quantization"),
        ("aggressive", "Compile + Quantization"),
    ]

    results = {}

    for config_name, config_desc in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config_desc}")
        print(f"{'='*70}")

        try:
            pipeline = OptimizedPipeline(optimization_level=config_name)
            pipeline.load()

            # Warmup run (first run is slower due to compilation)
            if config_name in ["compile", "aggressive"]:
                print("\nWarmup run (compilation happens here)...")
                pipeline.generate(prompt, seed=seed, warmup=True)
                print("✓ Warmup complete")

            # Actual timed runs
            print("\nTimed run:")
            image, elapsed = pipeline.generate(prompt, seed=seed)

            results[config_desc] = elapsed
            image.save(f"benchmark_{config_name}.png")
            print(f"✓ Saved: benchmark_{config_name}.png")

            # Clean up
            del pipeline
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Failed: {e}")
            results[config_desc] = None

    # Print results
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)

    baseline = results.get("Standard (baseline)")

    print(f"\n{'Configuration':<30} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 70)

    for config_desc, elapsed in results.items():
        if elapsed is not None:
            speedup = f"{baseline/elapsed:.2f}x" if baseline and baseline > 0 else "N/A"
            print(f"{config_desc:<30} {elapsed:<12.2f} {speedup:<10}")
        else:
            print(f"{config_desc:<30} {'FAILED':<12} {'N/A':<10}")

    print("\n" + "="*70)

    # Recommendation
    fastest = min(
        [(desc, time) for desc, time in results.items() if time is not None],
        key=lambda x: x[1],
        default=None
    )

    if fastest:
        print(f"✓ FASTEST: {fastest[0]} at {fastest[1]:.2f}s")
        print(f"\nRECOMMENDATION: Use '{fastest[0]}' configuration")
    else:
        print("❌ All optimizations failed")

    print("="*70)


if __name__ == "__main__":
    benchmark()
