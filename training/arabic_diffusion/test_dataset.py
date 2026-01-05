"""
Quick test script to verify synthetic Arabic dataset generation works.
Run this BEFORE training to ensure images are generated correctly.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import SyntheticArabicDataset

def test_dataset(arabic_words_file: str, output_dir: str = "test_samples"):
    """Test synthetic dataset generation."""
    
    # Load Arabic words
    if not os.path.exists(arabic_words_file):
        print(f"ERROR: Arabic words file not found: {arabic_words_file}")
        return False
    
    with open(arabic_words_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    if not words:
        print("ERROR: No Arabic words found in file!")
        return False
    
    print(f"✓ Loaded {len(words)} Arabic words")
    print(f"  Sample words: {words[:3]}")
    
    # Create dataset
    print("\nCreating synthetic dataset...")
    try:
        dataset = SyntheticArabicDataset(
            arabic_words=words,
            size=1024,
            num_samples=5,  # Just generate 5 samples for testing
        )
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Check fonts
        if dataset.font_paths:
            print(f"✓ Found {len(dataset.font_paths)} font(s): {dataset.font_paths}")
        else:
            print("⚠ WARNING: No Arabic fonts found! Images may not render correctly.")
            print("  Install Arabic fonts or provide font paths.")
        
    except Exception as e:
        print(f"ERROR creating dataset: {e}")
        return False
    
    # Generate and save test samples
    print("\nGenerating test samples...")
    os.makedirs(output_dir, exist_ok=True)
    
    from torchvision import transforms
    to_pil = transforms.ToPILImage()
    
    success_count = 0
    for i in range(min(5, len(dataset))):
        try:
            sample = dataset[i]
            word = sample['prompt']
            image_tensor = sample['pixel_values']
            
            # Denormalize: convert from [-1, 1] to [0, 1]
            image_tensor = (image_tensor + 1.0) / 2.0
            image_tensor = image_tensor.clamp(0, 1)
            
            # Convert to PIL and save
            pil_image = to_pil(image_tensor)
            save_path = os.path.join(output_dir, f"test_sample_{i:02d}_{word[:10]}.png")
            pil_image.save(save_path)
            
            print(f"  ✓ Generated sample {i+1}: {word} -> {save_path}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ ERROR generating sample {i+1}: {e}")
    
    print(f"\n{'='*60}")
    if success_count == 5:
        print("✓ SUCCESS: All test samples generated correctly!")
        print(f"  Check the '{output_dir}' folder to see the images.")
        print("  You can now proceed with training.")
        return True
    else:
        print(f"⚠ WARNING: Only {success_count}/5 samples generated successfully.")
        print("  Check the errors above and fix font/encoding issues.")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test synthetic Arabic dataset generation")
    parser.add_argument(
        "--arabic_words_file",
        type=str,
        required=True,
        help="Path to Arabic words file (one per line)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_samples",
        help="Directory to save test samples"
    )
    
    args = parser.parse_args()
    
    success = test_dataset(args.arabic_words_file, args.output_dir)
    sys.exit(0 if success else 1)

