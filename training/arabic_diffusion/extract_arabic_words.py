"""
Extract Arabic words from text files, websites, or generate from patterns.
This script helps build a large Arabic vocabulary.
"""

import os
import re
import sys
from pathlib import Path
from typing import Set, List
from collections import Counter


def extract_arabic_words_from_text(text: str, min_length: int = 2, max_length: int = 50) -> Set[str]:
    """
    Extract Arabic words from text.
    Arabic Unicode range: \u0600-\u06FF (includes Arabic, Persian, Urdu)
    """
    # Pattern for Arabic words (including diacritics)
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    words = set()
    
    for match in arabic_pattern.finditer(text):
        word = match.group().strip()
        # Remove diacritics for base word (optional)
        # word = re.sub(r'[\u064B-\u065F\u0670]', '', word)  # Remove diacritics
        
        if min_length <= len(word) <= max_length:
            # Remove common punctuation that might stick
            word = word.strip('.,;:!?()[]{}"\'-')
            if word:
                words.add(word)
    
    return words


def extract_from_file(file_path: str) -> Set[str]:
    """Extract Arabic words from a text file."""
    words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            words = extract_arabic_words_from_text(text)
        print(f"✓ Extracted {len(words)} words from {file_path}")
    except Exception as e:
        print(f"✗ Error reading {file_path}: {e}")
    return words


def extract_from_directory(directory: str, extensions: List[str] = None) -> Set[str]:
    """Extract Arabic words from all text files in a directory."""
    if extensions is None:
        extensions = ['.txt', '.md', '.py', '.json', '.xml', '.html']
    
    all_words = set()
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"✗ Directory not found: {directory}")
        return all_words
    
    file_count = 0
    for ext in extensions:
        for file_path in directory_path.rglob(f'*{ext}'):
            file_count += 1
            words = extract_from_file(str(file_path))
            all_words.update(words)
    
    print(f"✓ Processed {file_count} files, found {len(all_words)} unique words")
    return all_words


def generate_arabic_word_variations(base_words: Set[str]) -> Set[str]:
    """Generate variations of Arabic words with common prefixes and suffixes."""
    expanded = set(base_words)
    
    # Common Arabic prefixes
    prefixes = ["ال", "ب", "في", "من", "إلى", "على", "مع", "عن", "في", "ك", "ل"]
    
    # Common suffixes (simplified)
    suffixes = ["ة", "ين", "ات", "ون"]
    
    # Add prefixed versions
    for word in list(base_words)[:5000]:  # Limit to avoid explosion
        for prefix in prefixes:
            if not word.startswith(prefix) and len(word) > 2:
                expanded.add(prefix + word)
    
    return expanded


def load_existing_words(file_path: str) -> Set[str]:
    """Load existing words from file."""
    words = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and re.match(r'^[\u0600-\u06FF]+$', word):
                    words.add(word)
    return words


def save_words(words: Set[str], file_path: str, sort_by_length: bool = True):
    """Save words to file."""
    if sort_by_length:
        sorted_words = sorted(words, key=lambda x: (len(x), x))
    else:
        sorted_words = sorted(words)
    
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(word + '\n')
    
    print(f"✓ Saved {len(sorted_words)} unique words to {file_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract Arabic words from files or directories"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file or directory path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/arabic_words.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Expand words with prefixes/suffixes"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing words file"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Arabic Word Extractor")
    print("=" * 60)
    
    all_words = set()
    
    # Load existing words if merging
    if args.merge:
        existing = load_existing_words(args.output)
        all_words.update(existing)
        print(f"\nLoaded {len(existing)} existing words")
    
    # Extract from input
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            print(f"\nExtracting from file: {args.input}")
            words = extract_from_file(args.input)
            all_words.update(words)
        elif input_path.is_dir():
            print(f"\nExtracting from directory: {args.input}")
            words = extract_from_directory(args.input)
            all_words.update(words)
        else:
            print(f"✗ Input path not found: {args.input}")
            return
    else:
        print("\n⚠ No input specified. Use --input to specify file or directory.")
        print("   You can also manually add words to the output file.")
    
    # Expand if requested
    if args.expand and all_words:
        print(f"\nExpanding {len(all_words)} words with variations...")
        all_words = generate_arabic_word_variations(all_words)
        print(f"✓ Expanded to {len(all_words)} words")
    
    # Save
    if all_words:
        print(f"\nSaving {len(all_words)} unique words...")
        save_words(all_words, args.output)
        print("\n" + "=" * 60)
        print(f"✓ Complete! Created word list with {len(all_words)} unique words")
    else:
        print("\n⚠ No words extracted. Check your input path.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

