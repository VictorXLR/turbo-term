#!/usr/bin/env python3
"""
Terminal Image Viewer for Z-Image-Turbo
Displays generated images in Ghostty with interactive hotkeys
"""

import base64
import io
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import torch
from diffusers import ZImagePipeline
from PIL import Image


class TerminalImageViewer:
    OUTPUT_DIR = Path.home() / "Pictures" / "Autogen"

    def __init__(self):
        self.pipeline = None
        self.device = None
        self.dtype = None
        self.current_image = None
        self.current_prompt = ""
        self.last_prompt = ""
        self.temp_files = []
        # Ensure output directory exists
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def setup_device(self):
        """Setup device and dtype"""
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.bfloat16
            print("✓ Using Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
            print("✓ Using CUDA")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            print("⚠️  Using CPU (slower)")

    def load_model(self):
        """Load Z-Image-Turbo model"""
        if self.pipeline is not None:
            return

        print("📦 Loading Z-Image-Turbo model...")
        print("   First time will download ~32GB...")

        try:
            self.pipeline = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=self.dtype,
                use_safetensors=True,
            )
            self.pipeline = self.pipeline.to(self.device)

            # Enable memory optimizations
            if self.device in ["mps", "cuda"]:
                try:
                    self.pipeline.enable_attention_slicing()
                except:
                    pass

            print("✓ Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def generate_image(self, prompt: str, seed: int = 43) -> Optional[Image.Image]:
        """Generate image from prompt"""
        try:
            print(f"\n🎨 Generating: {prompt}")

            generator = torch.Generator(device=self.device).manual_seed(seed)

            image = self.pipeline(
                prompt=prompt,
                height=480,
                width=640,
                num_inference_steps=9,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]

            self.current_image = image
            self.current_prompt = prompt
            return image

        except Exception as e:
            print(f"❌ Error generating image: {e}")
            return None

    def display_image_kitty(self, image: Image.Image, max_width: int = 640) -> bool:
        """Display image using Kitty graphics protocol (works in Ghostty)"""
        try:
            # Resize for terminal display
            ratio = max_width / image.width
            display_image = image.resize(
                (max_width, int(image.height * ratio)),
                Image.Resampling.LANCZOS
            )

            # Convert to PNG bytes
            buf = io.BytesIO()
            display_image.save(buf, format="PNG")
            image_data = buf.getvalue()

            # Base64 encode
            b64_data = base64.standard_b64encode(image_data).decode("ascii")

            # Write using Kitty graphics protocol with chunking
            chunk_size = 4096
            chunks = [b64_data[i:i + chunk_size] for i in range(0, len(b64_data), chunk_size)]

            for i, chunk in enumerate(chunks):
                is_last = i == len(chunks) - 1
                if i == 0:
                    # First chunk: include all parameters
                    # a=T: transmit and display, f=100: PNG format, m=0/1: more data flag
                    cmd = f"\033_Ga=T,f=100,m={0 if is_last else 1};{chunk}\033\\"
                else:
                    # Continuation chunks
                    cmd = f"\033_Gm={0 if is_last else 1};{chunk}\033\\"
                sys.stdout.write(cmd)

            sys.stdout.write("\n")
            sys.stdout.flush()
            return True

        except Exception as e:
            print(f"Error displaying image: {e}")
            return False

    def display_image_chafa(self, image_path: str):
        """Fallback: Use chafa for ASCII/ANSI display"""
        try:
            result = subprocess.run(
                ["chafa", image_path, "--size=80x40", "--symbols=block"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(result.stdout)
                return True
        except FileNotFoundError:
            pass
        return False

    def display_image(self, image: Image.Image) -> str:
        """Display image in terminal and return temp file path"""
        # Save to temporary file for fallback/reference
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name
            self.temp_files.append(temp_path)

        # Try Kitty protocol first (Ghostty supports it)
        if not self.display_image_kitty(image):
            # Fallback to chafa if available
            if not self.display_image_chafa(temp_path):
                print(f"\n[Image saved to: {temp_path}]")
                print("(Install 'chafa' for better terminal display)")

        return temp_path

    def save_image(self, image: Image.Image, filename: str = None):
        """Save image to ~/Pictures/Autogen"""
        if filename is None:
            # Generate filename from prompt
            safe_prompt = "".join(
                c for c in self.current_prompt[:20] if c.isalnum() or c.isspace()
            ).rstrip()
            safe_prompt = safe_prompt.replace(" ", "_") or "output"
            counter = 1
            while True:
                filepath = self.OUTPUT_DIR / f"{safe_prompt}_{counter}.png"
                if not filepath.exists():
                    break
                counter += 1
        else:
            filepath = self.OUTPUT_DIR / filename

        image.save(filepath)
        print(f"✅ Image saved as: {filepath}")

    def copy_to_clipboard(self, image: Image.Image):
        """Copy image to clipboard (macOS)"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                # Use osascript to copy image to clipboard on macOS
                script = f'''
                    set theFile to POSIX file "{tmp.name}"
                    set theImage to read theFile as «class PNGf»
                    set the clipboard to theImage
                '''
                subprocess.run(["osascript", "-e", script], check=True)
                os.unlink(tmp.name)
            print("✅ Image copied to clipboard")
        except Exception as e:
            print(f"❌ Failed to copy to clipboard: {e}")

    def delete_image(self, image_path: str):
        """Delete temporary image file"""
        try:
            os.unlink(image_path)
            print("🗑️  Image deleted")
        except:
            pass

    def show_menu(self, image_path: str):
        """Show interactive menu"""
        print(f"\n" + "=" * 60)
        print(f"🖼️  Z-Image-Turbo Terminal Viewer")
        print(f"Prompt: {self.current_prompt}")
        print("=" * 60)
        print("\nControls:")
        print("  [s] Save image")
        print("  [c] Copy to clipboard")
        print("  [d] Delete image")
        print("  [r] Retry same prompt")
        print("  [n] New prompt")
        print("  [q] Quit")
        print("-" * 60)

        while True:
            try:
                choice = input("\n> ").lower().strip()

                if choice == "s":
                    filename = input(
                        "Enter filename (or press Enter for auto): "
                    ).strip()
                    self.save_image(self.current_image, filename if filename else None)

                elif choice == "c":
                    self.copy_to_clipboard(self.current_image)

                elif choice == "d":
                    self.delete_image(image_path)
                    return "delete"

                elif choice == "r":
                    return "retry"

                elif choice == "n":
                    self.last_prompt = self.current_prompt
                    return "new"

                elif choice == "q":
                    return "quit"

                else:
                    print("Invalid choice. Try: s, c, d, r, n, q")

            except KeyboardInterrupt:
                return "quit"

    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

    def run(self):
        """Main interactive loop"""
        self.setup_device()
        self.load_model()

        try:
            while True:
                # Get prompt
                if not self.current_prompt:
                    prompt = input("\nprompt > ").strip()
                else:
                    prompt = self.current_prompt

                if not prompt:
                    continue

                # Generate image
                image = self.generate_image(prompt)
                if image is None:
                    continue

                # Display image
                image_path = self.display_image(image)

                # Show menu and handle choice
                action = self.show_menu(image_path)

                if action == "quit":
                    break
                elif action == "new":
                    self.current_prompt = ""
                    self.current_image = None
                elif action == "delete":
                    self.current_image = None
                    self.current_prompt = ""
                elif action == "retry":
                    # Keep same prompt, generate new image
                    continue

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        finally:
            self.cleanup()


if __name__ == "__main__":
    viewer = TerminalImageViewer()
    viewer.run()
