"""
Modal Compute Integration for Model Evaluation

This module provides Modal app and image configuration for running evaluations on cloud GPUs.
"""

import modal
from typing import List, Dict
from pathlib import Path

# Create Modal app
app = modal.App("mental-health-evaluation")

# Get local directory for copying files
local_dir = Path(__file__).parent

# Define container image with all dependencies for running local models
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "datasets",
        "openai",
        "sentencepiece",
        "protobuf",
        "google-generativeai",
        "anthropic",
        "langchain-openai",
        "python-dotenv",
    )
    .add_local_dir(local_dir, remote_path="/root/code")
)


# Define Modal functions at global scope for Mistral and Llama2 evaluation
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    cpu=4.0,
    memory=32768,
    secrets=[modal.Secret.from_name("openai-secret")],
)
def evaluate_mistral_modal_remote(questions: List[Dict], judge: str = "gpt4") -> List[Dict]:
    """Evaluate Mistral on Modal GPU - runs remotely."""
    import sys
    sys.path.insert(0, "/root/code")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.evaluation_part1 import evaluate_base_mistral

    print("\nLoading Mistral-7B-Instruct on Modal GPU...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✓ Mistral loaded on Modal GPU")

    return evaluate_base_mistral(questions, model, tokenizer, judge=judge)


# CLI helper to test Modal connection
@app.function(image=image, gpu="T4")
def test_gpu():
    """Test function to verify Modal GPU access."""
    import torch

    has_cuda = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if has_cuda else 0
    device_name = torch.cuda.get_device_name(0) if has_cuda else "No GPU"

    return {
        "pytorch_version": torch.__version__,
        "cuda_available": has_cuda,
        "gpu_count": device_count,
        "gpu_name": device_name,
    }


@app.local_entrypoint()
def test():
    """Test Modal setup - run with: modal run modal_compute.py"""
    print("Testing Modal GPU access...")
    result = test_gpu.remote()

    print("\n✅ Modal GPU Test Results:")
    print(f"   PyTorch: {result['pytorch_version']}")
    print(f"   CUDA Available: {result['cuda_available']}")
    print(f"   GPU Count: {result['gpu_count']}")
    print(f"   GPU Name: {result['gpu_name']}")

    if result['cuda_available']:
        print("\n🚀 Modal is ready for GPU workloads!")
    else:
        print("\n⚠️  Warning: GPU not detected")
