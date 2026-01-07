#!/usr/bin/env python
# Quick verification script to check if production training is ready

import os
from pathlib import Path

print("=" * 60)
print("Production Training Setup Verification")
print("=" * 60)

# Check 1: Vocabulary file
vocab_file = Path("Output/arabic_words.txt")
if vocab_file.exists():
    with open(vocab_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    print(f"✅ Vocabulary file found: {len(words)} words")
    if len(words) >= 10000:
        print(f"   ✓ Large vocabulary ({len(words)} words) - Good for production!")
    else:
        print(f"   ⚠️  Small vocabulary ({len(words)} words) - Consider expanding")
else:
    print(f"❌ Vocabulary file NOT found: {vocab_file}")
    print(f"   Expected location: training/arabic_diffusion/Output/arabic_words.txt")

# Check 2: Training script
train_script = Path("train_lora.py")
if train_script.exists():
    print(f"✅ Training script found: {train_script}")
else:
    print(f"❌ Training script NOT found: {train_script}")

# Check 3: Dataset module
dataset_file = Path("dataset.py")
if dataset_file.exists():
    print(f"✅ Dataset module found: {dataset_file}")
    
    # Check if it has production features
    with open(dataset_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'use_english_prompts' in content:
            print(f"   ✓ Production features enabled (English prompts)")
        if 'composition_mode' in content:
            print(f"   ✓ Production features enabled (Mixed composition)")
else:
    print(f"❌ Dataset module NOT found: {dataset_file}")

# Check 4: Output directory
output_dir = Path("arabic-lora-production")
if output_dir.exists():
    print(f"⚠️  Output directory exists: {output_dir}")
    print(f"   (Will overwrite or create new checkpoints)")
else:
    print(f"✅ Output directory ready: {output_dir} (will be created)")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if vocab_file.exists() and len(words) >= 10000:
    print("✅ READY FOR PRODUCTION TRAINING!")
    print("\nNext step: Run production training:")
    print("  python train_lora.py --arabic_words_file Output/arabic_words.txt \\")
    print("    --output_dir arabic-lora-production --use_english_prompts \\")
    print("    --composition_mode mixed [other params...]")
else:
    print("❌ NOT READY - Fix issues above first")

print("=" * 60)

