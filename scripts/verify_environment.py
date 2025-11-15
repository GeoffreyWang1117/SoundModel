#!/usr/bin/env python3
"""
Environment verification script.
Checks that all required libraries are installed and GPUs are accessible.
"""

import sys


def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

        return True
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False


def check_transformers():
    """Check transformers library."""
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Transformers not found: {e}")
        return False


def check_tts():
    """Check TTS library."""
    try:
        import TTS
        print(f"✓ TTS (Coqui) installed")
        return True
    except ImportError as e:
        print(f"✗ TTS not found: {e}")
        return False


def check_audio_libs():
    """Check audio processing libraries."""
    libs = {
        "librosa": "librosa",
        "soundfile": "soundfile",
        "audioread": "audioread",
    }

    all_ok = True
    for name, module in libs.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError as e:
            print(f"✗ {name} not found: {e}")
            all_ok = False

    return all_ok


def check_other_deps():
    """Check other important dependencies."""
    libs = {
        "accelerate": "accelerate",
        "peft": "peft",
        "datasets": "datasets",
        "bitsandbytes": "bitsandbytes",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
        "yaml": "yaml",
        "omegaconf": "omegaconf",
    }

    all_ok = True
    for name, module in libs.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError as e:
            print(f"✗ {name} not found: {e}")
            all_ok = False

    return all_ok


def main():
    """Run all checks."""
    print("=" * 60)
    print("Audio-Augmented LLM Environment Verification")
    print("=" * 60)

    print("\n[1] Checking PyTorch and CUDA...")
    pytorch_ok = check_pytorch()

    print("\n[2] Checking Transformers...")
    transformers_ok = check_transformers()

    print("\n[3] Checking TTS...")
    tts_ok = check_tts()

    print("\n[4] Checking Audio Processing Libraries...")
    audio_ok = check_audio_libs()

    print("\n[5] Checking Other Dependencies...")
    other_ok = check_other_deps()

    print("\n" + "=" * 60)
    if all([pytorch_ok, transformers_ok, tts_ok, audio_ok, other_ok]):
        print("✓ All checks passed! Environment is ready.")
        return 0
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print("\nTo install all dependencies, run:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
