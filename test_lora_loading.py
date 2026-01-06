"""
Diagnostic script to test if Arabic LoRA is loading correctly.
"""

import os
import torch
from diffusers import StableDiffusion3Pipeline

def test_lora_loading(lora_path):
    """Test if LoRA loads correctly."""
    print("=" * 60)
    print("Arabic LoRA Loading Diagnostic")
    print("=" * 60)
    
    # Check if path exists
    print(f"\n1. Checking LoRA path: {lora_path}")
    if not os.path.isabs(lora_path):
        # Try relative paths
        possible_paths = [
            lora_path,
            os.path.join("training", "arabic_diffusion", lora_path),
            os.path.join("arabic-lora-output", lora_path),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                lora_path = os.path.abspath(path)
                print(f"   ✓ Found at: {lora_path}")
                break
        else:
            print(f"   ✗ Path not found in any location")
            return False
    else:
        if not os.path.exists(lora_path):
            print(f"   ✗ Path does not exist")
            return False
        print(f"   ✓ Path exists")
    
    # Check for safetensors file
    safetensors_path = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
    if os.path.exists(safetensors_path):
        file_size = os.path.getsize(safetensors_path) / (1024 * 1024)  # MB
        print(f"   ✓ Found safetensors file: {safetensors_path}")
        print(f"   ✓ File size: {file_size:.2f} MB")
    else:
        print(f"   ✗ Safetensors file not found: {safetensors_path}")
        return False
    
    # Try loading pipeline
    print(f"\n2. Loading SD3 pipeline...")
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float32
        )
        print("   ✓ Pipeline loaded")
    except Exception as e:
        print(f"   ✗ Failed to load pipeline: {e}")
        return False
    
    # Try loading LoRA
    print(f"\n3. Loading LoRA weights...")
    try:
        pipe.load_lora_weights(lora_path, adapter_name="arabic")
        print("   ✓ LoRA weights loaded successfully")
        
        # Check active adapters
        if hasattr(pipe, 'get_active_adapters'):
            adapters = pipe.get_active_adapters()
            print(f"   ✓ Active adapters: {adapters}")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed to load LoRA: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    # Default path
    lora_path = "training/arabic_diffusion/arabic-lora-output"
    
    if len(sys.argv) > 1:
        lora_path = sys.argv[1]
    
    success = test_lora_loading(lora_path)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ LoRA loading test PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ LoRA loading test FAILED")
        print("=" * 60)
        print("\nPossible issues:")
        print("1. LoRA path is incorrect")
        print("2. LoRA file is corrupted")
        print("3. LoRA format is incompatible")
        print("4. Missing dependencies")
    
    sys.exit(0 if success else 1)

