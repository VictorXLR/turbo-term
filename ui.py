#!/usr/bin/env python3
"""
Z-Image-Turbo Gradio UI
Web interface for image generation with Z-Image-Turbo
"""

import time
from typing import Optional

import gradio as gr
import torch
from diffusers import ZImagePipeline


class ImageGenerator:
    """Handles model loading and image generation"""

    def __init__(self):
        self.pipeline = None
        self.device = None
        self.dtype = None

    def load_model(self):
        """Load the model once and cache it"""
        if self.pipeline is not None:
            return "Model already loaded"

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

        print(f"Loading model on {device_info}...")

        # Load pipeline
        self.pipeline = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=self.dtype,
            use_safetensors=True,
            trust_remote_code=True,
        )

        self.pipeline = self.pipeline.to(self.device)

        # Enable memory optimizations
        if self.device in ["mps", "cuda"]:
            try:
                self.pipeline.enable_attention_slicing()
            except:
                pass

        return f"Model loaded successfully on {device_info}"

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

            info = f"Generated in {elapsed:.2f}s | Resolution: {width}x{height} | Steps: {num_steps}"
            if seed is not None and seed >= 0:
                info += f" | Seed: {seed}"

            return image, info

        except Exception as e:
            return None, f"Error during generation: {str(e)}"


# Initialize generator
generator = ImageGenerator()


def create_ui():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Z-Image-Turbo") as demo:
        gr.Markdown(
            """
            # Z-Image-Turbo Image Generator
            Generate high-quality images with sub-second inference using Z-Image-Turbo.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Model loading
                load_btn = gr.Button("Load Model", variant="primary", size="lg")
                load_status = gr.Textbox(
                    label="Model Status",
                    value="Model not loaded - Click 'Load Model' to start",
                    interactive=False,
                )

                gr.Markdown("### Generation Settings")

                # Prompt inputs
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a lovely cat sitting on a windowsill, photorealistic, detailed, 4k",
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
                        value=-1,
                        precision=0,
                        info="Set to -1 for random, or use specific number for reproducibility",
                    )

                # Generate button
                generate_btn = gr.Button("Generate Image", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="Generated Image", type="pil")
                generation_info = gr.Textbox(label="Generation Info", interactive=False)

        # Event handlers
        load_btn.click(fn=generator.load_model, outputs=load_status)

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
                    "a cute robot reading a book in a cozy library, digital art",
                    "blurry, bad anatomy, low quality",
                ],
                [
                    "an astronaut riding a horse on the moon, cinematic lighting",
                    "blurry, distorted, low quality",
                ],
            ],
            inputs=[prompt, negative_prompt],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True to create a public link
    )
