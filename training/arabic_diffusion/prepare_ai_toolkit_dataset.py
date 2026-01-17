#!/usr/bin/env python
# coding=utf-8
"""
Pre-generate dataset for Ostris AI Toolkit training.

This script generates images and captions in the format required by AI Toolkit:
- Images: PNG/JPG files
- Captions: Text files with matching names (e.g., image_0001.png -> image_0001.txt)
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional

try:
    from .dataset import SyntheticArabicDataset
except ImportError:
    from dataset import SyntheticArabicDataset


def load_arabic_words(file_path: str) -> List[str]:
    """Load Arabic words from file (one per line, or word frequency format)."""
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle word frequency format: "word 123456" -> extract just "word"
            word = line.split()[0] if line.split() else line
            if word:
                words.append(word)
    return words


def generate_dataset(
    arabic_words_file: str,
    output_dir: str,
    num_samples: int,
    image_size: int = 1024,
    composition_mode: str = "mixed",
    max_phrase_length: int = 5,
    character_level_prob: float = 0.1,
    use_english_prompts: bool = True,
):
    """
    Generate dataset for AI Toolkit.
    
    Args:
        arabic_words_file: Path to file with Arabic words
        output_dir: Directory to save images and captions
        num_samples: Number of samples to generate
        image_size: Image resolution
        composition_mode: "word", "phrase", "character", or "mixed"
        max_phrase_length: Maximum words in a phrase
        character_level_prob: Probability of character-level samples
        use_english_prompts: Use English descriptions in captions
    """
    # Load Arabic words
    print(f"Loading Arabic words from: {arabic_words_file}")
    arabic_words = load_arabic_words(arabic_words_file)
    print(f"Loaded {len(arabic_words)} Arabic words")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset instance (we'll use it to generate images)
    # Note: We'll use the dataset's methods directly, not __getitem__
    dataset = SyntheticArabicDataset(
        arabic_words=arabic_words,
        size=image_size,
        num_samples=num_samples,
        use_english_prompts=use_english_prompts,
        composition_mode=composition_mode,
        max_phrase_length=max_phrase_length,
        character_level_prob=character_level_prob,
    )
    
    print(f"\nGenerating {num_samples} samples...")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Composition mode: {composition_mode}")
    print(f"Using English prompts: {use_english_prompts}\n")
    
    # Generate samples
    # We need to render images directly since dataset returns tensors
    
    for i in range(num_samples):
        # Compose Arabic text
        arabic_text = dataset._compose_text()
        
        # Choose random style parameters
        font_path = random.choice(dataset.font_paths) if dataset.font_paths else None
        font_size = random.choice(dataset.font_sizes)
        bg_color = random.choice(dataset.background_colors)
        text_color = random.choice(dataset.text_colors)
        
        # Render image
        image = dataset._render_arabic_text(arabic_text, font_path, font_size, text_color, bg_color)
        
        # Generate prompt
        if dataset.use_english_prompts:
            prompt_template = random.choice(dataset.prompt_templates)
            prompt = prompt_template.format(text=arabic_text)
        else:
            prompt = arabic_text
        
        # Save image
        image_filename = f"image_{i:06d}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)
        
        # Save caption (AI Toolkit expects .txt file with same name)
        caption_filename = f"image_{i:06d}.txt"
        caption_path = os.path.join(output_dir, caption_filename)
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")
    
    print(f"\n✓ Dataset generation complete!")
    print(f"  - Images: {num_samples} PNG files")
    print(f"  - Captions: {num_samples} TXT files")
    print(f"  - Location: {output_dir}")
    print(f"\nNext step: Use this dataset with AI Toolkit for training.")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate dataset for Ostris AI Toolkit training"
    )
    
    parser.add_argument(
        "--arabic_words_file",
        type=str,
        required=True,
        help="Path to file containing Arabic words (one per line).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images and captions.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to generate. Default: 10000",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image resolution. Default: 1024",
    )
    parser.add_argument(
        "--composition_mode",
        type=str,
        default="mixed",
        choices=["word", "phrase", "character", "mixed"],
        help="Text composition mode. Default: 'mixed'",
    )
    parser.add_argument(
        "--max_phrase_length",
        type=int,
        default=5,
        help="Maximum number of words in a phrase. Default: 5",
    )
    parser.add_argument(
        "--character_level_prob",
        type=float,
        default=0.1,
        help="Probability of character-level samples in mixed mode. Default: 0.1",
    )
    parser.add_argument(
        "--use_english_prompts",
        action="store_true",
        default=True,
        help="Use English descriptions in captions (recommended). Default: True",
    )
    
    args = parser.parse_args()
    
    # Validate
    if not os.path.exists(args.arabic_words_file):
        raise ValueError(f"Arabic words file not found: {args.arabic_words_file}")
    
    # Generate dataset
    generate_dataset(
        arabic_words_file=args.arabic_words_file,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        image_size=args.image_size,
        composition_mode=args.composition_mode,
        max_phrase_length=args.max_phrase_length,
        character_level_prob=args.character_level_prob,
        use_english_prompts=args.use_english_prompts,
    )


if __name__ == "__main__":
    main()
