#!/usr/bin/env python3
"""
Z-Image-Turbo Optimized Gradio UI
Web interface with performance optimizations for M4 Max / Apple Silicon
"""

import time
from typing import Optional

import gradio as gr
import torch
from diffusers import ZImagePipeline


class OptimizedImageGenerator:
    """Handles model loading and image generation with optimizations"""

    def __init__(self):
        self.pipeline = None
        self.device = None
        self.dtype = None
        self.optimization_level = "standard"

    def load_model(self, optimization: str = "Standard"):
        """Load the model once and cache it with selected optimization"""
        if self.pipeline is not None:
            return f"Model already loaded with {self.optimization_level} optimization. Reload page to change."

        # Map UI choice to internal level
        opt_map = {
            "Standard": "standard",
            "Torch Compile (Recommended for M4)": "compile",
            "Int8 Quantization": "quantize",
            "Aggressive (Compile + Quantize)": "aggressive"
        }
        self.optimization_level = opt_map.get(optimization, "standard")

        # Determine device
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.bfloat16
            device_info = "Metal Performance Shaders (MPS) - Apple Silicon"
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
            device_info = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            device_info = "CPU (slower)"

        print(f"Loading model on {device_info} with {optimization}...")

        try:
            # Load pipeline
            self.pipeline = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=self.dtype,
                use_safetensors=True,
            )

            self.pipeline = self.pipeline.to(self.device)

            # Apply optimizations
            self._apply_optimizations()

            # Warmup for compiled models
            if self.optimization_level in ["compile", "aggressive"]:
                print("Running warmup for compiled model...")
                self._warmup()

            return f"✓ Model loaded on {device_info}\n✓ Optimization: {optimization}\n✓ Ready to generate"

        except Exception as e:
            return f"❌ Error loading model: {e}"

    def _apply_optimizations(self):
        """Apply selected optimization level"""
        # Always enable attention slicing
        try:
            self.pipeline.enable_attention_slicing()
            print("✓ Attention slicing enabled")
        except:
            pass

        if self.optimization_level in ["quantize", "aggressive"]:
            try:
                from torchao.quantization import quantize_, int8_weight_only

                print("Applying int8 quantization...")
                if hasattr(self.pipeline, "transformer"):
                    quantize_(self.pipeline.transformer, int8_weight_only())
                elif hasattr(self.pipeline, "unet"):
                    quantize_(self.pipeline.unet, int8_weight_only())
                print("✓ Model quantized to int8")
            except Exception as e:
                print(f"⚠ Quantization failed: {e}")

        if self.optimization_level in ["compile", "aggressive"]:
            try:
                print("Compiling model (first run will be slower)...")
                if hasattr(self.pipeline, "transformer"):
                    self.pipeline.transformer = torch.compile(
                        self.pipeline.transformer,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                elif hasattr(self.pipeline, "unet"):
                    self.pipeline.unet = torch.compile(
                        self.pipeline.unet,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                print("✓ Model compiled")
            except Exception as e:
                print(f"⚠ Compilation failed: {e}")

    def _warmup(self):
        """Warmup run for compiled models"""
        try:
            generator = torch.Generator(device=self.device).manual_seed(42)
            _ = self.pipeline(
                prompt="warmup",
                height=512,
                width=512,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=generator,
            )
            print("✓ Warmup complete")
        except:
            pass

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_steps: int,
        seed: Optional[int],
        progress=gr.Progress(),
    ):
        """Generate an image from the prompt"""
        if self.pipeline is None:
            return None, "Error: Model not loaded. Click 'Load Model' first."

        if not prompt.strip():
            return None, "Error: Please enter a prompt"

        # Prepare generator for reproducibility
        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            progress(0, desc="Generating image...")
            start_time = time.time()

            # Generate image
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=0.0,  # Turbo models use 0.0
                generator=generator,
            ).images[0]

            elapsed = time.time() - start_time

            info = f"⚡ {elapsed:.2f}s | {width}x{height} | {num_steps} steps | {self.optimization_level}"
            if seed is not None and seed >= 0:
                info += f" | seed: {seed}"

            return image, info

        except Exception as e:
            return None, f"Error during generation: {str(e)}"


# Initialize generator
generator = OptimizedImageGenerator()


def create_ui():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Z-Image-Turbo Optimized") as demo:
        gr.Markdown(
            """
            # Z-Image-Turbo (Optimized for M4 Max)
            Generate high-quality images with performance optimizations.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Model loading with optimization selection
                gr.Markdown("### Model Loading")

                optimization_choice = gr.Radio(
                    choices=[
                        "Standard",
                        "Torch Compile (Recommended for M4)",
                        "Int8 Quantization",
                        "Aggressive (Compile + Quantize)"
                    ],
                    value="Torch Compile (Recommended for M4)",
                    label="Optimization Level",
                    info="Torch Compile gives 2-3x speedup on M4. First generation is slower (compilation)."
                )

                load_btn = gr.Button("Load Model", variant="primary", size="lg")
                load_status = gr.Textbox(
                    label="Model Status",
                    value="Model not loaded - Select optimization and click 'Load Model'",
                    interactive=False,
                    lines=3,
                )

                gr.Markdown("### Generation Settings")

                # Prompt inputs
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a serene mountain landscape at sunset, photorealistic, 8k",
                    lines=3,
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, distorted, ugly, bad anatomy",
                    lines=2,
                )

                # Resolution settings
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=1024,
                    )

                # Quick resolution presets
                with gr.Row():
                    gr.Markdown("**Presets:**")
                preset_square = gr.Button("1024x1024", size="sm")
                preset_landscape = gr.Button("1280x720", size="sm")
                preset_portrait = gr.Button("720x1280", size="sm")

                # Advanced settings
                with gr.Accordion("Advanced Settings", open=True):
                    num_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=4,
                        maximum=16,
                        step=1,
                        value=9,
                        info="More steps = better quality (recommended: 9)",
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=43,
                        precision=0,
                        info="Set to -1 for random. Some seeds (like 42) may produce black images.",
                    )

                # Generate button
                generate_btn = gr.Button(
                    "Generate Image", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="Generated Image", type="pil")
                generation_info = gr.Textbox(
                    label="Generation Info",
                    interactive=False,
                    lines=2
                )

                gr.Markdown(
                    """
                    ### Performance Tips
                    - **First generation** with Torch Compile is slow (20-60s) - this is compilation
                    - **Subsequent generations** are 2-3x faster (~15-20s on M4 Max)
                    - Lower resolution = faster (try 768x768 for quick iterations)
                    - Quantization trades quality for speed
                    """
                )

        # Event handlers
        load_btn.click(
            fn=generator.load_model,
            inputs=[optimization_choice],
            outputs=load_status
        )

        generate_btn.click(
            fn=generator.generate,
            inputs=[prompt, negative_prompt, width, height, num_steps, seed],
            outputs=[output_image, generation_info],
        )

        # Preset buttons
        preset_square.click(lambda: (1024, 1024), outputs=[width, height])
        preset_landscape.click(lambda: (1280, 720), outputs=[width, height])
        preset_portrait.click(lambda: (720, 1280), outputs=[width, height])

        # Examples
        gr.Examples(
            examples=[
                [
                    "a serene mountain landscape at sunset, photorealistic, 8k",
                    "blurry, low quality, distorted",
                ],
                [
                    "a futuristic cyberpunk city with neon lights, highly detailed",
                    "blurry, low quality, ugly",
                ],
                [
                    "portrait of a woman with natural makeup, hyperrealistic, 4k",
                    "blurry, distorted, low quality, bad anatomy",
                ],
            ],
            inputs=[prompt, negative_prompt],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
