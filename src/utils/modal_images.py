"""
Pre-built Modal images for mental health research.

Run this once to build and cache the image:
    modal build modal_images.py

Then import and use in other scripts.
"""

import modal

# Create a persistent image that gets cached
mental_health_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML libraries
        "torch==2.1.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "bitsandbytes==0.41.3",
        "peft==0.7.0",

        # Data and utilities
        "datasets==2.16.0",
        "sentencepiece==0.1.99",
        "protobuf==4.25.1",

        # OpenAI for judge
        "openai==1.6.0",
    )
    # Optional: Add model downloads to the image itself for instant startup
    # .run_commands(
    #     "python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\"mistralai/Mistral-7B-Instruct-v0.2\")'",
    #     "python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\"unsloth/Llama-3.2-1B-Instruct\")'",
    # )
)

app = modal.App("mental-health-images")

# Test function to verify image works
@app.function(image=mental_health_image, gpu="T4")
def test_image():
    """Test that all packages are installed correctly."""
    import torch
    import transformers
    import peft
    from openai import OpenAI

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    }


@app.local_entrypoint()
def main():
    """Test the pre-built image."""
    print("Testing pre-built image...")
    result = test_image.remote()

    print("\n✅ Image test results:")
    for key, value in result.items():
        print(f"   {key}: {value}")

    print("\n🎉 Image is ready to use! Import it in your evaluation scripts.")
