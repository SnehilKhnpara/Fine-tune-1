"""
Script to download and expand Arabic word lists from various sources.
This helps create a comprehensive Arabic vocabulary for training.
"""

import os
import sys
import requests
import re
from pathlib import Path
from typing import List, Set
from urllib.parse import urlparse


def download_file(url: str, output_path: str) -> bool:
    """Download a file from URL."""
    try:
        print(f"Downloading from {url}...")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        return False


def extract_words_from_text(text: str) -> Set[str]:
    """Extract Arabic words from text."""
    # Arabic Unicode range: \u0600-\u06FF
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    words = set()
    
    for match in arabic_pattern.finditer(text):
        word = match.group().strip()
        # Filter out very short words and very long words
        if 2 <= len(word) <= 50:
            words.add(word)
    
    return words


def load_existing_words(file_path: str) -> Set[str]:
    """Load existing words from file."""
    words = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    words.add(word)
    return words


def save_words(words: Set[str], file_path: str):
    """Save words to file, one per line."""
    sorted_words = sorted(words, key=lambda x: (len(x), x))
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(word + '\n')
    print(f"✓ Saved {len(sorted_words)} unique words to {file_path}")


def generate_common_arabic_words() -> Set[str]:
    """Generate a comprehensive list of common Arabic words."""
    words = set()
    
    # Common greetings and phrases
    greetings = [
        "مرحبا", "شكرا", "سلام", "أهلا", "مع السلامة",
        "صباح الخير", "مساء الخير", "السلام عليكم", "كيف الحال",
        "أهلا وسهلا", "بارك الله فيك", "جزاك الله خيرا"
    ]
    
    # Days of week
    days = [
        "السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"
    ]
    
    # Months
    months = [
        "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
        "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"
    ]
    
    # Numbers (1-100)
    numbers = [
        "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة",
        "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون", "مائة"
    ]
    
    # Colors
    colors = [
        "أحمر", "أزرق", "أخضر", "أصفر", "أسود", "أبيض", "رمادي", "بني", "وردي", "برتقالي"
    ]
    
    # Common verbs (infinitive form)
    verbs = [
        "ذهب", "جاء", "أكل", "شرب", "نام", "استيقظ", "قرأ", "كتب", "درس", "عمل",
        "لعب", "ركض", "مشى", "تكلم", "سمع", "رأى", "فهم", "علم", "أحب", "كره"
    ]
    
    # Common nouns
    nouns = [
        "كتاب", "قلم", "ورقة", "ماء", "خبز", "لحم", "فاكهة", "خضار",
        "بيت", "سيارة", "طائرة", "سفينة", "بحر", "صحراء", "جبل",
        "شمس", "قمر", "نجوم", "سماء", "أرض", "نهر", "شجرة", "زهرة",
        "طائر", "قطة", "كلب", "حصان", "جمل", "أسد", "فيل"
    ]
    
    # Family
    family = [
        "أب", "أم", "أخ", "أخت", "جد", "جدة", "عم", "عمة", "خال", "خالة",
        "ابن", "ابنة", "حفيد", "حفيدة", "زوج", "زوجة"
    ]
    
    # Professions
    professions = [
        "طبيب", "معلم", "مهندس", "محامي", "طالب", "طاهي", "نجار", "حداد",
        "خياط", "بناء", "سباك", "كهربائي", "ممرض", "شرطي", "بائع", "موظف"
    ]
    
    # Places
    places = [
        "مدينة", "قرية", "شارع", "سوق", "متجر", "مستشفى", "مطار", "ميناء",
        "حديقة", "ملعب", "مكتبة", "مطعم", "فندق", "مدرسة", "جامعة"
    ]
    
    # Combine all
    all_words = greetings + days + months + numbers + colors + verbs + nouns + family + professions + places
    
    for word in all_words:
        words.add(word)
    
    return words


def download_arabic_corpus_words() -> Set[str]:
    """Try to download Arabic words from online sources."""
    words = set()
    
    # Option 1: Try to download from a public Arabic word list
    # Note: These URLs may need to be updated with actual sources
    sources = [
        # Add actual URLs to Arabic word lists here
        # Example: "https://example.com/arabic-words.txt"
    ]
    
    for url in sources:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                downloaded_words = extract_words_from_text(response.text)
                words.update(downloaded_words)
                print(f"✓ Extracted {len(downloaded_words)} words from {url}")
        except Exception as e:
            print(f"✗ Could not download from {url}: {e}")
    
    return words


def expand_with_combinations(words: Set[str], max_combinations: int = 10000) -> Set[str]:
    """Generate word combinations and common phrases."""
    expanded = set(words)
    word_list = list(words)
    
    if len(word_list) < 2:
        return expanded
    
    # Common prefixes
    prefixes = ["ال", "ب", "في", "من", "إلى", "على", "مع"]
    
    # Add words with common prefixes
    for word in word_list[:1000]:  # Limit to avoid too many combinations
        for prefix in prefixes:
            if not word.startswith(prefix):
                expanded.add(prefix + word)
    
    # Common two-word phrases (limited)
    common_phrases = [
        "السلام عليكم", "صباح الخير", "مساء الخير", "مع السلامة",
        "أهلا وسهلا", "كيف الحال", "بارك الله فيك"
    ]
    
    for phrase in common_phrases:
        expanded.add(phrase)
    
    return expanded


def main():
    """Main function to build Arabic word list."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and expand Arabic word list")
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/arabic_words.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Expand with combinations and prefixes"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=10000,
        help="Target number of unique words"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Arabic Word List Builder")
    print("=" * 60)
    
    # Load existing words
    existing_words = load_existing_words(args.output_file)
    print(f"\nLoaded {len(existing_words)} existing words from {args.output_file}")
    
    # Generate common words
    print("\n1. Generating common Arabic words...")
    common_words = generate_common_arabic_words()
    print(f"   ✓ Generated {len(common_words)} common words")
    
    # Try to download from online sources
    print("\n2. Attempting to download from online sources...")
    downloaded_words = download_arabic_corpus_words()
    if downloaded_words:
        print(f"   ✓ Downloaded {len(downloaded_words)} words")
    else:
        print("   ⚠ No online sources available (you can add URLs to the script)")
    
    # Combine all words
    all_words = existing_words | common_words | downloaded_words
    
    # Expand if requested
    if args.expand:
        print("\n3. Expanding with combinations and prefixes...")
        all_words = expand_with_combinations(all_words)
        print(f"   ✓ Expanded to {len(all_words)} words")
    
    # Save
    print(f"\n4. Saving {len(all_words)} unique words...")
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    save_words(all_words, args.output_file)
    
    print("\n" + "=" * 60)
    print(f"✓ Complete! Created word list with {len(all_words)} unique words")
    print(f"  File: {args.output_file}")
    print("\nNote: The training script will generate millions of training samples")
    print("      from this word list by randomly selecting and rendering them.")
    print("=" * 60)


if __name__ == "__main__":
    main()

