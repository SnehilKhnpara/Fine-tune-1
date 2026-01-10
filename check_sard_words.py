#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check if SARD words file is correct (complete words, not fragments).
"""

import re
import sys

def analyze_words_file(filename: str):
    """Analyze the words file for correctness."""
    print(f"Analyzing: {filename}")
    print("=" * 60)
    
    words = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                words.append(word)
    
    if not words:
        print("❌ File is empty!")
        return False
    
    print(f"Total words: {len(words)}")
    
    # Analyze word lengths
    base_lengths = []
    has_punctuation = []
    fragments_2char = []
    valid_words = []
    
    for word in words:
        # Remove diacritics for counting
        base_chars = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', word)
        base_lengths.append(len(base_chars))
        
        # Check for punctuation (shouldn't be in words)
        if re.search(r'[،,\.\?\!;:\(\)\[\]\{\}]', word):
            has_punctuation.append(word)
        
        # Check if 2-character fragment
        if len(base_chars) == 2:
            fragments_2char.append(word)
        elif len(base_chars) >= 3:
            valid_words.append(word)
    
    # Statistics
    avg_length = sum(base_lengths) / len(base_lengths) if base_lengths else 0
    min_length = min(base_lengths) if base_lengths else 0
    max_length = max(base_lengths) if base_lengths else 0
    
    print(f"\nWord Length Statistics:")
    print(f"  - Average base characters: {avg_length:.2f}")
    print(f"  - Minimum: {min_length} chars")
    print(f"  - Maximum: {max_length} chars")
    print(f"  - Words with 2 base chars: {len(fragments_2char)} ({len(fragments_2char)/len(words)*100:.1f}%)")
    print(f"  - Words with 3+ base chars: {len(valid_words)} ({len(valid_words)/len(words)*100:.1f}%)")
    
    # Check for issues
    issues = []
    
    if len(fragments_2char) > len(words) * 0.1:  # More than 10% are 2-char
        issues.append(f"⚠️  Too many 2-character fragments: {len(fragments_2char)} words")
        print(f"\n⚠️  WARNING: {len(fragments_2char)} words are 2-character fragments")
        print(f"   Sample fragments: {fragments_2char[:10]}")
    
    if has_punctuation:
        issues.append(f"⚠️  Words with punctuation: {len(has_punctuation)} words")
        print(f"\n⚠️  WARNING: {len(has_punctuation)} words contain punctuation")
        print(f"   Sample: {has_punctuation[:10]}")
    
    # Show sample words
    print(f"\nSample words (first 30):")
    for i, word in enumerate(words[:30], 1):
        base_len = len(re.sub(r'[\u064B-\u065F\u0670\u0640]', '', word))
        has_punct = "⚠️" if re.search(r'[،,\.\?\!;:\(\)\[\]\{\}]', word) else "✓"
        print(f"  {i:3d}. {has_punct} {word} ({base_len} base chars)")
    
    # Overall assessment
    print(f"\n{'=' * 60}")
    if len(valid_words) >= len(words) * 0.8:  # At least 80% are 3+ chars
        print("✓ File looks GOOD - mostly complete words (3+ characters)")
        if issues:
            print(f"\nMinor issues found:")
            for issue in issues:
                print(f"  {issue}")
        return True
    else:
        print("❌ File has ISSUES - too many fragments (2-character words)")
        print(f"   Recommendation: Run clean_sard_words.py to filter fragments")
        return False

if __name__ == "__main__":
    filename = "sard_words.txt"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    try:
        is_good = analyze_words_file(filename)
        sys.exit(0 if is_good else 1)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
