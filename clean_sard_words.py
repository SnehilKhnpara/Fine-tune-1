#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean and filter SARD extracted words to remove fragments.
Fixes the issue where words are split into 2-character fragments.
"""

import re
from typing import Set, List


def is_valid_complete_word(word: str, min_base_chars: int = 3) -> bool:
    """
    Check if a word is a complete Arabic word, not a fragment.
    
    Args:
        word: Word to check
        min_base_chars: Minimum number of base characters (default: 3)
    
    Returns:
        True if valid complete word, False if fragment
    """
    if not word or len(word.strip()) == 0:
        return False
    
    word = word.strip()
    
    # Remove diacritics and tatweel for counting base characters
    base_chars = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', word)
    
    # Must have minimum base characters (3+ to avoid 2-char fragments)
    if len(base_chars) < min_base_chars:
        return False
    
    # Exclude single repeated characters
    if re.match(r'^[\u0600-\u06FF]\1+$', word):
        return False
    
    # Allow common short words even if 2-3 chars (these are real words)
    common_short_words = {
        'ال', 'من', 'في', 'عن', 'على', 'إلى', 'مع', 'عن', 'هل', 'أن', 
        'إن', 'أو', 'بل', 'لك', 'له', 'لها', 'لهما', 'لهم', 'لهن',
        'الله', 'محمد', 'القرآن', 'الإسلام'
    }
    
    if word in common_short_words:
        return True
    
    # Exclude obvious fragments (single prefixes/suffixes)
    fragment_patterns = [
        r'^[بفكلمنت]ـ?$',  # Single prefix characters with optional tatweel
        r'^[أإآ]?[بفكلمنت]ـ?$',  # Prefix with hamza
    ]
    
    for pattern in fragment_patterns:
        if re.match(pattern, word):
            return False
    
    # Exclude if it's just a diacritic or punctuation
    if re.match(r'^[\u064B-\u065F\u0670]+$', word):
        return False
    
    return True


def clean_sard_words(input_file: str, output_file: str, min_length: int = 3):
    """
    Clean SARD words file by removing fragments.
    
    Args:
        input_file: Path to input sard_words.txt
        output_file: Path to output cleaned file
        min_length: Minimum base character length (default: 3)
    """
    print(f"Cleaning words from: {input_file}")
    print(f"Minimum word length: {min_length} base characters")
    print("=" * 60)
    
    # Read all words
    all_words = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                all_words.add(word)
    
    print(f"Loaded {len(all_words)} words from input file")
    
    # Filter words
    valid_words = set()
    fragments_filtered = 0
    
    for word in all_words:
        if is_valid_complete_word(word, min_base_chars=min_length):
            valid_words.add(word)
        else:
            fragments_filtered += 1
    
    # Sort words (by length, then alphabetically)
    sorted_words = sorted(valid_words, key=lambda x: (len(re.sub(r'[\u064B-\u065F\u0670\u0640]', '', x)), x))
    
    # Save cleaned words
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(word + '\n')
    
    print(f"\nResults:")
    print(f"  - Original words: {len(all_words)}")
    print(f"  - Fragments filtered: {fragments_filtered}")
    print(f"  - Valid words: {len(sorted_words)}")
    print(f"  - Saved to: {output_file}")
    
    # Show statistics
    if sorted_words:
        base_lengths = [len(re.sub(r'[\u064B-\u065F\u0670\u0640]', '', w)) for w in sorted_words]
        print(f"\nStatistics:")
        print(f"  - Average word length: {sum(base_lengths) / len(base_lengths):.2f} base characters")
        print(f"  - Shortest word: {min(sorted_words, key=lambda x: len(re.sub(r'[\u064B-\u065F\u0670\u0640]', '', x)))} ({min(base_lengths)} chars)")
        print(f"  - Longest word: {max(sorted_words, key=lambda x: len(re.sub(r'[\u064B-\u065F\u0670\u0640]', '', x)))} ({max(base_lengths)} chars)")
        
        print(f"\nSample words (first 30):")
        for i, word in enumerate(sorted_words[:30], 1):
            base_len = len(re.sub(r'[\u064B-\u065F\u0670\u0640]', '', word))
            print(f"  {i:2d}. {word} ({base_len} base chars)")
        
        if len(sorted_words) > 30:
            print(f"  ... and {len(sorted_words) - 30} more")


if __name__ == "__main__":
    import sys
    
    input_file = "sard_words.txt"
    output_file = "sard_words_cleaned.txt"
    min_length = 3
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        min_length = int(sys.argv[3])
    
    try:
        clean_sard_words(input_file, output_file, min_length)
        print(f"\n✓ Successfully cleaned words!")
        print(f"  Use '{output_file}' for training instead of '{input_file}'")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
