#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract Arabic words from SARD-Extended HuggingFace dataset.
"""

import os
import re
import sys
from pathlib import Path
from typing import Set
from tqdm import tqdm

# Try to import datasets with better error handling
try:
    from datasets import load_dataset
except ImportError as e:
    print("ERROR: datasets library not installed.")
    print("Please install it: pip install datasets")
    sys.exit(1)
except (AttributeError, ModuleNotFoundError) as e:
    error_msg = str(e)
    if "PyExtensionType" in error_msg or "pyarrow" in error_msg.lower():
        print("=" * 60)
        print("ERROR: Version incompatibility between 'datasets' and 'pyarrow'")
        print("=" * 60)
        print("\nTo fix this, please run:")
        print("  pip install --upgrade pyarrow")
        print("  pip install --upgrade datasets")
        print("\nOr try:")
        print("  pip install pyarrow>=10.0.0 datasets>=2.14.0")
        print("\nOr use the fix script: fix_dependencies.bat")
        print("=" * 60)
        sys.exit(1)
    else:
        print(f"ERROR importing datasets: {e}")
        print("\nTry: pip install --upgrade datasets pyarrow")
        sys.exit(1)


def extract_arabic_words(text: str, min_word_length: int = 3) -> Set[str]:
    """
    Extract complete Arabic words from text.
    Filters out fragments and ensures we get complete words.
    
    Args:
        text: Input text (may contain Arabic and other characters)
        min_word_length: Minimum length of base Arabic characters (default: 3)
    
    Returns:
        Set of complete Arabic words
    """
    if not text:
        return set()
    
    # Pattern to match Arabic words (Arabic characters + optional diacritics)
    # This matches sequences of Arabic characters separated by non-Arabic
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    
    # Find all Arabic sequences
    words = arabic_pattern.findall(text)
    
    filtered_words = set()
    for word in words:
        word = word.strip()
        if not word:
            continue
        
        # Remove diacritics for length counting (keep only base characters)
        base_chars = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', word)  # Remove diacritics and tatweel
        
        # Filter criteria:
        # 1. Minimum length: at least min_word_length base characters (avoid fragments)
        # 2. Maximum length: reasonable word length (avoid concatenated text)
        # 3. Must contain at least one base Arabic character (not just diacritics)
        if len(base_chars) >= min_word_length and len(word) <= 50:
            # Additional check: word should not look like a fragment
            # (e.g., single repeated character, or obvious split pattern)
            # Fix: Use capturing group before referencing it
            if not re.match(r'^([\u0600-\u06FF])\1+$', word):  # Not single char repeated
                filtered_words.add(word)
    
    return filtered_words


def is_valid_arabic_word(word: str, min_length: int = 3) -> bool:
    """
    Check if a word is a valid complete Arabic word (not a fragment).
    
    Args:
        word: Arabic word to check
        min_length: Minimum base character length
    
    Returns:
        True if valid word, False if fragment
    """
    if not word or len(word.strip()) == 0:
        return False
    
    word = word.strip()
    
    # Remove diacritics for counting
    base_chars = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', word)
    
    # Must meet minimum length
    if len(base_chars) < min_length:
        return False
    
    # Exclude single repeated characters (fix: use capturing group)
    if re.match(r'^([\u0600-\u06FF])\1+$', word):
        return False
    
    # Allow common short words even if 2-3 chars (these are real words)
    common_short_words = {
        'ال', 'من', 'في', 'عن', 'على', 'إلى', 'مع', 'هل', 'أن', 
        'إن', 'أو', 'بل', 'لك', 'له', 'الله', 'محمد'
    }
    
    if word in common_short_words:
        return True
    
    return True


def combine_chunks_and_extract_words(chunk_text: str) -> Set[str]:
    """
    Combine chunk text and extract complete words.
    Handles cases where words are split across chunks.
    
    Args:
        chunk_text: Text from chunk field (may contain multiple chunks/lines)
    
    Returns:
        Set of complete Arabic words
    """
    if not chunk_text:
        return set()
    
    # Replace common separators with spaces to help word extraction
    # Keep Arabic text together, separate from other characters
    normalized = re.sub(r'[^\u0600-\u06FF\s]', ' ', chunk_text)  # Replace non-Arabic/non-space with space
    normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
    
    # Extract words (will handle space-separated words correctly)
    words = extract_arabic_words(normalized, min_word_length=3)
    
    return words


def load_sard_dataset_all_splits(max_samples_per_split: int = None):
    """
    Load SARD-Extended dataset from HuggingFace.
    SARD-Extended is organized by fonts, not train/test splits.
    
    Args:
        max_samples_per_split: Maximum samples per font split (None = all)
    
    Returns:
        List of datasets (one per font split)
    """
    print("Loading SARD-Extended dataset...")
    try:
        # Load all splits (fonts)
        all_datasets = load_dataset("riotu-lab/SARD-Extended")
        
        print(f"✓ Found {len(all_datasets)} font splits: {list(all_datasets.keys())}")
        
        datasets_list = []
        for font_name, dataset in all_datasets.items():
            print(f"  - {font_name}: {len(dataset)} samples")
            if max_samples_per_split:
                dataset = dataset.select(range(min(max_samples_per_split, len(dataset))))
                print(f"    Using {len(dataset)} samples")
            datasets_list.append((font_name, dataset))
        
        return datasets_list
    except Exception as e:
        print(f"ERROR: Could not load SARD-Extended dataset: {e}")
        return None


def extract_words_from_sample(sample, text_field: str = "chunk"):
    """
    Extract Arabic words from a single sample.
    
    Args:
        sample: Single dataset sample (dict)
        text_field: Name of the field containing text (default: "chunk" for SARD)
    
    Returns:
        Set of Arabic words from this sample
    """
    words = set()
    
    # Try to get text from the specified field
    text = None
    if text_field in sample:
        text = sample[text_field]
    
    # If not found, try common alternatives (in priority order)
    if not text or len(text) == 0:
        # SARD-Extended specific: try 'chunk' first
        alternatives = ["chunk", "text", "label", "transcription", "gt", "ground_truth", "content", "words"]
        for alt in alternatives:
            if alt in sample:
                value = sample[alt]
                if isinstance(value, str) and len(value) > 0:
                    text = value
                    text_field = alt
                    break
    
    # If still not found, try any string field (except known non-text fields)
    if not text:
        exclude = ['image_name', 'font_name', 'image_base64']
        for key, value in sample.items():
            if key not in exclude and isinstance(value, str) and len(value) > 0:
                # Check if it might contain text (has some characters beyond just names)
                if len(value) > 3 or re.search(r'[\u0600-\u06FF]', value):
                    text = value
                    text_field = key
                    break
    
    if text:
        words = extract_arabic_words(text)
    
    return words, text_field


def extract_words_from_dataset(dataset, font_name: str = "", text_field: str = "chunk", max_samples: int = None):
    """
    Extract all unique Arabic words from a dataset.
    
    Args:
        dataset: HuggingFace dataset
        font_name: Name of the font split (for logging)
        text_field: Name of the field containing text
        max_samples: Maximum number of samples to process (None = all)
    
    Returns:
        Set of unique Arabic words
    """
    all_words = set()
    
    desc = f"Processing {font_name}" if font_name else "Processing samples"
    
    # Check what fields are available
    if len(dataset) > 0:
        sample = dataset[0]
        available_fields = list(sample.keys())
        print(f"  Available fields: {available_fields}")
        
        # SARD-Extended has 'chunk' field which should contain the Arabic text
        # For SARD-Extended, always use 'chunk' field (it's the standard field for text)
        # Check multiple samples to verify chunk contains Arabic
        print(f"  Inspecting 'chunk' field in first 10 samples...")
        chunk_with_arabic = 0
        chunk_preview = None
        
        if 'chunk' in available_fields:
            for i in range(min(10, len(dataset))):
                test_sample = dataset[i]
                chunk_val = test_sample.get('chunk')
                if chunk_val and isinstance(chunk_val, str) and len(chunk_val) > 0:
                    if re.search(r'[\u0600-\u06FF]', chunk_val):
                        chunk_with_arabic += 1
                        if chunk_preview is None:
                            chunk_preview = chunk_val[:100] if len(chunk_val) > 100 else chunk_val
            
            if chunk_with_arabic > 0:
                text_field = 'chunk'
                print(f"  ✓ Using field: 'chunk' (found Arabic in {chunk_with_arabic}/10 samples)")
                if chunk_preview:
                    print(f"    Preview: {repr(chunk_preview)}")
            else:
                text_field = 'chunk'  # Use chunk anyway - might have content later
                print(f"  → Using field: 'chunk' (field exists, may have Arabic in other samples)")
        else:
            # Fallback: try to find any field with Arabic
            exclude_fields = ['image_name', 'font_name', 'image_base64']
            for key in available_fields:
                if key not in exclude_fields:
                    value = sample.get(key)
                    if isinstance(value, str) and len(value) > 0 and re.search(r'[\u0600-\u06FF]', value):
                        text_field = key
                        print(f"  ✓ Using field: '{text_field}' (contains Arabic text)")
                        break
            else:
                text_field = 'chunk'  # Default to chunk
                print(f"  ⚠️  Using field: 'chunk' (default, inspect samples manually if no results)")
    
    # Limit samples if specified
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    # Force use 'chunk' field for SARD-Extended (it's the text field)
    if 'chunk' in available_fields:
        text_field = 'chunk'
        print(f"  Using 'chunk' field for SARD-Extended")
        print(f"  Minimum word length: 3 base characters (filtering 2-char fragments)")
    
    processed = 0
    empty_chunks = 0
    chunks_with_arabic = 0
    chunks_with_content = 0
    total_fragments_filtered = 0
    
    # For SARD, chunks might be fragments - combine and extract complete words
    for sample in tqdm(dataset, desc=desc, leave=False):
        processed += 1
        # Directly get chunk value
        chunk_val = sample.get(text_field, '')
        
        # Track statistics
        if not chunk_val or len(chunk_val) == 0:
            empty_chunks += 1
        else:
            chunks_with_content += 1
            if isinstance(chunk_val, str) and re.search(r'[\u0600-\u06FF]', chunk_val):
                chunks_with_arabic += 1
                
                # IMPORTANT: SARD chunks might be fragments, so we need to:
                # 1. Combine multiple chunks if needed (for now, process each chunk)
                # 2. Extract words with minimum length of 3 base characters
                # 3. Filter out obvious fragments
                
                # Normalize text: replace non-Arabic characters with spaces to help word extraction
                normalized = re.sub(r'[^\u0600-\u06FF\s]', ' ', chunk_val)
                normalized = re.sub(r'\s+', ' ', normalized).strip()
                
                # Extract words (minimum 3 base characters)
                words = extract_arabic_words(normalized, min_word_length=3)
                
                # Additional validation: filter out obvious fragments
                valid_words = set()
                for word in words:
                    if is_valid_arabic_word(word, min_length=3):
                        valid_words.add(word)
                    else:
                        total_fragments_filtered += 1
                
                all_words.update(valid_words)
    
    # Report statistics
    print(f"\n  Processing statistics:")
    print(f"    - Total samples: {processed}")
    print(f"    - Chunks with content: {chunks_with_content}")
    print(f"    - Chunks with Arabic: {chunks_with_arabic}")
    print(f"    - Empty chunks: {empty_chunks}")
    print(f"    - Fragments filtered out: {total_fragments_filtered}")
    print(f"    - Valid unique words found: {len(all_words)}")
    print(f"    - Minimum word length: 3 base characters")
    
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
        default=None,
        help="NOT USED: SARD-Extended is organized by fonts, not train/test splits. All fonts will be processed."
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
    max_samples_per_split = args.max_samples
    split = args.split  # Not used for SARD (it's organized by fonts)
    
    # Test mode: limit to 500 samples per font
    if args.test:
        max_samples_per_split = 500
        print("🧪 TEST MODE: Processing only 500 samples per font")
    
    # Load all font splits
    font_datasets = load_sard_dataset_all_splits(max_samples_per_split=max_samples_per_split)
    if font_datasets is None or len(font_datasets) == 0:
        print("\n❌ Failed to load dataset. Exiting.")
        return
    
    # Extract words from all font splits
    print(f"\n{'=' * 60}")
    print(f"Extracting Arabic words from {len(font_datasets)} font splits...")
    print(f"{'=' * 60}\n")
    
    all_words = set()
    total_samples_processed = 0
    
    for font_name, dataset in font_datasets:
        dataset_size = len(dataset)
        print(f"\nProcessing font: {font_name} ({dataset_size} samples)")
        words = extract_words_from_dataset(
            dataset, 
            font_name=font_name, 
            max_samples=max_samples_per_split
        )
        all_words.update(words)
        total_samples_processed += dataset_size
        print(f"  ✓ Found {len(words)} unique words from this font")
        print(f"  ✓ Total unique words so far: {len(all_words)}")
    
    arabic_words = all_words
    total_samples = total_samples_processed
    
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
    print(f"\n{'=' * 60}")
    print(f"Statistics:")
    print(f"{'=' * 60}")
    print(f"  Total samples processed: {total_samples}")
    print(f"  Total unique words: {len(sorted_words)}")
    if sorted_words:
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
