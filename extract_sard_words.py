#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract Arabic words from SARD-Extended HuggingFace dataset.
"""

import os
import re
from pathlib import Path
from typing import Set
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets library not installed.")
    print("Please install it: pip install datasets")
    exit(1)


def extract_arabic_words(text: str) -> Set[str]:
    """
    Extract Arabic words from text.
    Arabic Unicode range: U+0600 to U+06FF
    """
    if not text:
        return set()
    
    # Pattern to match Arabic words (Arabic characters + optional diacritics)
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    
    # Find all Arabic words
    words = arabic_pattern.findall(text)
    
    # Filter: remove very short words (likely diacritics only) and very long ones
    filtered_words = set()
    for word in words:
        # Remove diacritics for counting (keep only base characters)
        base_chars = re.sub(r'[\u064B-\u065F\u0670]', '', word)  # Remove diacritics
        if len(base_chars) >= 2 and len(word) <= 50:  # Reasonable word length
            filtered_words.add(word.strip())
    
    return filtered_words


def load_sard_dataset(split: str = "train", max_samples: int = None):
    """
    Load SARD-Extended dataset from HuggingFace.
    
    Args:
        split: Dataset split to load ("train", "test", etc.)
        max_samples: Maximum number of samples to process (None = all)
    
    Returns:
        Dataset object
    """
    print(f"Loading SARD-Extended dataset (split: {split})...")
    try:
        dataset = load_dataset("riotu-lab/SARD-Extended", split=split)
        print(f"✓ Loaded {len(dataset)} samples")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            print(f"✓ Using {len(dataset)} samples")
        
        return dataset
    except Exception as e:
        print(f"ERROR: Could not load SARD-Extended dataset: {e}")
        print("\nTrying alternative approach...")
        try:
            # Try loading without specifying split
            dataset = load_dataset("riotu-lab/SARD-Extended")
            if split in dataset:
                dataset = dataset[split]
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                print(f"✓ Loaded {len(dataset)} samples")
                return dataset
            else:
                print(f"ERROR: Split '{split}' not found. Available splits: {list(dataset.keys())}")
                return None
        except Exception as e2:
            print(f"ERROR: Failed to load dataset: {e2}")
            return None


def extract_words_from_dataset(dataset, text_field: str = "text"):
    """
    Extract all unique Arabic words from the dataset.
    
    Args:
        dataset: HuggingFace dataset
        text_field: Name of the field containing text
    
    Returns:
        Set of unique Arabic words
    """
    all_words = set()
    
    print(f"\nExtracting Arabic words from dataset...")
    print(f"Looking for text in field: '{text_field}'")
    
    # Check what fields are available
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Available fields: {list(sample.keys())}")
        
        # Try to find text field
        if text_field not in sample:
            # Try common alternatives
            alternatives = ["text", "label", "transcription", "gt", "ground_truth", "content"]
            for alt in alternatives:
                if alt in sample:
                    text_field = alt
                    print(f"Using field: '{text_field}'")
                    break
            else:
                print(f"WARNING: '{text_field}' field not found. Trying all string fields...")
    
    processed = 0
    for sample in tqdm(dataset, desc="Processing samples"):
        processed += 1
        
        # Try to get text from various possible fields
        text = None
        if text_field in sample:
            text = sample[text_field]
        else:
            # Try to find any string field
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 0:
                    text = value
                    break
        
        if text:
            words = extract_arabic_words(text)
            all_words.update(words)
        
        # Progress update every 1000 samples
        if processed % 1000 == 0:
            print(f"  Processed {processed} samples, found {len(all_words)} unique words so far...")
    
    return all_words


def main():
    """Main function to extract and save Arabic words."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract Arabic words from SARD-Extended dataset")
    parser.add_argument(
        "--output_file",
        type=str,
        default="sard_words.txt",
        help="Output file path (default: sard_words.txt)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None = all, use small number for testing)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 1000 samples"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SARD-Extended Arabic Word Extractor")
    print("=" * 60)
    
    # Configuration
    output_file = args.output_file
    max_samples = args.max_samples
    split = args.split
    
    # Test mode: limit to 1000 samples
    if args.test:
        max_samples = 1000
        print("🧪 TEST MODE: Processing only 1000 samples")
    
    # Load dataset
    dataset = load_sard_dataset(split=split, max_samples=max_samples)
    if dataset is None:
        print("\n❌ Failed to load dataset. Exiting.")
        return
    
    # Extract words
    arabic_words = extract_words_from_dataset(dataset)
    
    if not arabic_words:
        print("\n⚠️  No Arabic words found in dataset.")
        print("This might mean:")
        print("  - The dataset structure is different than expected")
        print("  - The text field name is different")
        print("  - The dataset doesn't contain Arabic text in the expected format")
        return
    
    # Sort words for better readability
    sorted_words = sorted(arabic_words, key=lambda x: (len(x), x))
    
    # Save to file
    print(f"\n{'=' * 60}")
    print(f"Saving {len(sorted_words)} unique Arabic words to: {output_file}")
    print(f"{'=' * 60}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(word + '\n')
    
    print(f"\n✓ Successfully saved {len(sorted_words)} Arabic words to {output_file}")
    print(f"\nFile size: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    # Show some statistics
    print(f"\nStatistics:")
    print(f"  Total unique words: {len(sorted_words)}")
    print(f"  Average word length: {sum(len(w) for w in sorted_words) / len(sorted_words):.2f} characters")
    print(f"  Shortest word: {min(sorted_words, key=len)} ({len(min(sorted_words, key=len))} chars)")
    print(f"  Longest word: {max(sorted_words, key=len)} ({len(max(sorted_words, key=len))} chars)")
    
    # Show sample words
    print(f"\nSample words (first 20):")
    for i, word in enumerate(sorted_words[:20], 1):
        print(f"  {i:2d}. {word}")
    
    if len(sorted_words) > 20:
        print(f"  ... and {len(sorted_words) - 20} more")


if __name__ == "__main__":
    import sys
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
