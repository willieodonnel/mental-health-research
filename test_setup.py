"""
Quick test script to verify your environment is set up correctly
for running the mental health inference pipeline.
"""

import sys

def check_python_version():
    """Check Python version."""
    print("[*] Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   [OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   [FAIL] Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_torch():
    """Check PyTorch installation."""
    print("\n[*] Checking PyTorch...")
    try:
        import torch
        print(f"   [OK] PyTorch {torch.__version__} installed")
        return True
    except ImportError:
        print("   [FAIL] PyTorch not installed")
        print("   Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

def check_cuda():
    """Check CUDA availability."""
    print("\n[*] Checking CUDA/GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   [OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   [OK] CUDA version: {torch.version.cuda}")
            print(f"   [OK] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("   [WARN] CUDA not available (will run on CPU - very slow)")
            print("   Make sure you installed PyTorch with CUDA support")
            return False
    except:
        return False

def check_transformers():
    """Check transformers library."""
    print("\n[*] Checking Transformers...")
    try:
        import transformers
        print(f"   [OK] Transformers {transformers.__version__} installed")
        return True
    except ImportError:
        print("   [FAIL] Transformers not installed")
        print("   Install: pip install transformers")
        return False

def check_bitsandbytes():
    """Check bitsandbytes for quantization."""
    print("\n[*] Checking bitsandbytes (for quantization)...")
    try:
        import bitsandbytes
        print(f"   [OK] bitsandbytes installed")
        return True
    except ImportError:
        print("   [FAIL] bitsandbytes not installed")
        print("   Install: pip install bitsandbytes")
        return False

def check_accelerate():
    """Check accelerate library."""
    print("\n[*] Checking accelerate...")
    try:
        import accelerate
        print(f"   [OK] accelerate {accelerate.__version__} installed")
        return True
    except ImportError:
        print("   [FAIL] accelerate not installed")
        print("   Install: pip install accelerate")
        return False

def estimate_memory():
    """Estimate memory requirements."""
    print("\n[*] Memory Estimation:")
    print("   4-bit quantization: ~3.5-4 GB VRAM")
    print("   8-bit quantization: ~7 GB VRAM")
    print("   16-bit (half precision): ~14 GB VRAM")
    print("   32-bit (full precision): ~28 GB VRAM")

    try:
        import torch
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n   Your GPU has {total_mem:.1f} GB VRAM")

            if total_mem >= 16:
                print("   [OK] Plenty of memory for all quantization levels!")
            elif total_mem >= 8:
                print("   [OK] Good for 4-bit and 8-bit quantization")
            elif total_mem >= 6:
                print("   [WARN] Recommended: Use 4-bit quantization only")
            else:
                print("   [FAIL] May not have enough VRAM for inference")
    except:
        pass

def main():
    """Run all checks."""
    print("="*60)
    print("Mental Health Inference Pipeline - Setup Check")
    print("="*60)

    checks = [
        check_python_version(),
        check_torch(),
        check_cuda(),
        check_transformers(),
        check_bitsandbytes(),
        check_accelerate(),
    ]

    estimate_memory()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    passed = sum(checks)
    total = len(checks)

    print(f"\nPassed: {passed}/{total} checks")

    if passed == total:
        print("\n[OK] All checks passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("  1. Run: python mental_health_inference.py")
        print("  2. Check INFERENCE_README.md for more details")
    elif passed >= 4:
        print("\n[WARN] Most checks passed. Review warnings above.")
        print("   You may be able to run on CPU (will be slow)")
    else:
        print("\n[FAIL] Several checks failed. Install missing dependencies:")
        print("   pip install -r requirements_inference.txt")

    # Quick CUDA test
    print("\n" + "="*60)
    print("Quick CUDA Test")
    print("="*60)

    try:
        import torch
        if torch.cuda.is_available():
            print("\nCreating small tensor on GPU...")
            x = torch.randn(100, 100).cuda()
            y = x @ x.T
            print("[OK] GPU computation successful!")
            print(f"   Result shape: {y.shape}")
            print(f"   Memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        else:
            print("\n[WARN] CUDA not available, skipping GPU test")
    except Exception as e:
        print(f"\n[FAIL] GPU test failed: {e}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
