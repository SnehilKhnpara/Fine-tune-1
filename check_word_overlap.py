"""Check which words from prompts are in the training data."""
import re
from pathlib import Path

# Load training words
train_file = Path("data/arabic_words.txt")
prompts_file = Path("prompts_arabic.txt")

train_words = set()
with open(train_file, 'r', encoding='utf-8') as f:
    for line in f:
        word = line.strip()
        if word:
            train_words.add(word)

print(f"Training words: {len(train_words)}")

# Extract words from prompts
prompt_words = set()
prompt_lines = []
with open(prompts_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        prompt_lines.append(line)
        # Extract Arabic words from prompt
        arabic_words = re.findall(r'[\u0600-\u06FF]+', line)
        for word in arabic_words:
            prompt_words.add(word)

print(f"Unique words in prompts: {len(prompt_words)}")

# Find matching words
matching_words = train_words & prompt_words
missing_words = prompt_words - train_words

print(f"\n✓ Words in BOTH training and prompts: {len(matching_words)}")
print(f"✗ Words in prompts but NOT in training: {len(missing_words)}")

if matching_words:
    print(f"\nFirst 20 matching words:")
    for word in sorted(matching_words)[:20]:
        print(f"  - {word}")

if missing_words:
    print(f"\nFirst 20 missing words:")
    for word in sorted(missing_words)[:20]:
        print(f"  - {word}")

# Create demo prompt file with only matching words
demo_prompts = []
for line in prompt_lines:
    arabic_words_in_line = re.findall(r'[\u0600-\u06FF]+', line)
    # Check if all Arabic words in this prompt are in training data
    if arabic_words_in_line and all(word in train_words for word in arabic_words_in_line):
        demo_prompts.append(line)

print(f"\n{'='*60}")
print(f"Creating demo_prompts_matching.txt with {len(demo_prompts)} prompts")
print(f"  (only prompts where ALL Arabic words are in training data)")
print(f"{'='*60}")

with open("demo_prompts_matching.txt", 'w', encoding='utf-8') as f:
    for prompt in demo_prompts:
        f.write(prompt + '\n')

print(f"✓ Saved to: demo_prompts_matching.txt")
