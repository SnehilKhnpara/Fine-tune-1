"""
Dataset classes for Arabic diffusion fine-tuning.

Supports:
1. Synthetic Arabic scene text dataset (primary)
2. EvArEST dataset (secondary, for OCR evaluation)
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display


class SyntheticArabicDataset(Dataset):
    """
    Generates synthetic Arabic text images on simple backgrounds.
    
    PRODUCTION-READY: Supports character-level, word-level, and phrase-level generation.
    
    Features:
    - Clean Arabic text rendered on simple backgrounds (words, phrases, or characters)
    - Multiple fonts, sizes, colors
    - Correct RTL and glyph connections
    - Generated programmatically
    - English prompt templates for better model understanding
    - Character composition for arbitrary Arabic text generation
    """
    
    def __init__(
        self,
        arabic_words: List[str],
        size: int = 1024,
        num_samples: int = 10000,
        font_paths: Optional[List[str]] = None,
        background_colors: Optional[List[Tuple[int, int, int]]] = None,
        text_colors: Optional[List[Tuple[int, int, int]]] = None,
        font_sizes: Optional[List[int]] = None,
        use_english_prompts: bool = True,  # Production: Use English descriptions
        prompt_templates: Optional[List[str]] = None,
        composition_mode: str = "mixed",  # "word", "phrase", "character", "mixed"
        max_phrase_length: int = 5,  # Max words in a phrase
        character_level_prob: float = 0.1,  # Probability of character-level samples
    ):
        self.arabic_words = arabic_words
        self.size = size
        self.num_samples = num_samples
        self.use_english_prompts = use_english_prompts
        self.composition_mode = composition_mode
        self.max_phrase_length = max_phrase_length
        self.character_level_prob = character_level_prob
        
        # Extract unique Arabic characters for character-level training
        self.arabic_chars = self._extract_arabic_characters(arabic_words)
        
        # English prompt templates for production use
        if prompt_templates is None:
            self.prompt_templates = [
                "A poster with the Arabic text '{text}'",
                "A billboard displaying the Arabic phrase '{text}'",
                "A coffee mug with the Arabic text '{text}'",
                "A street sign saying '{text}' in Arabic",
                "A vintage sign reading '{text}' in Arabic",
                "A journal cover with the Arabic phrase '{text}'",
                "A store window with an Arabic sign '{text}'",
                "A t-shirt with the Arabic print '{text}'",
                "A book cover titled '{text}' in Arabic",
                "A welcome mat saying '{text}' in Arabic",
                "A wall poster with the Arabic words '{text}'",
                "A greeting card that says '{text}' in Arabic",
                "A shop board advertising '{text}' in Arabic",
                "A banner with the Arabic message '{text}'",
                "A photo of a notepad with '{text}' in Arabic",
                "A neon sign reading '{text}' in Arabic",
                "A street mural with the Arabic quote '{text}'",
                "A restaurant menu with '{text}' in Arabic",
                "A chalkboard sign saying '{text}' in Arabic",
                "A framed print with the Arabic words '{text}'",
            ]
        else:
            self.prompt_templates = prompt_templates
        
        # Default fonts (user should provide Arabic fonts)
        self.font_paths = font_paths or []
        if not self.font_paths:
            # Try to find system Arabic fonts
            self._find_arabic_fonts()
        
        # Default colors
        self.background_colors = background_colors or [
            (255, 255, 255),  # White
            (240, 240, 240),  # Light gray
            (250, 250, 250),  # Off-white
        ]
        self.text_colors = text_colors or [
            (0, 0, 0),        # Black
            (50, 50, 50),    # Dark gray
            (20, 20, 20),    # Very dark gray
        ]
        self.font_sizes = font_sizes or [32, 40, 48, 56, 64]
    
    def _extract_arabic_characters(self, words: List[str]) -> List[str]:
        """Extract unique Arabic characters for character-level training."""
        chars = set()
        for word in words:
            for char in word:
                # Check if character is Arabic (Unicode range 0600-06FF)
                if '\u0600' <= char <= '\u06FF':
                    chars.add(char)
        return sorted(list(chars))
    
    def _compose_text(self) -> str:
        """
        Compose Arabic text based on composition_mode.
        Production-ready: Can generate words, phrases, or characters.
        """
        if self.composition_mode == "character":
            # Character-level: single Arabic character
            return random.choice(self.arabic_chars)
        elif self.composition_mode == "word":
            # Word-level: single word
            return random.choice(self.arabic_words)
        elif self.composition_mode == "phrase":
            # Phrase-level: multiple words
            num_words = random.randint(2, self.max_phrase_length)
            words = random.sample(self.arabic_words, min(num_words, len(self.arabic_words)))
            return " ".join(words)
        else:  # "mixed" - production mode
            # Mixed: randomly choose character, word, or phrase
            rand = random.random()
            if rand < self.character_level_prob:
                # Character-level sample
                return random.choice(self.arabic_chars)
            elif rand < 0.7:
                # Word-level sample (60% of non-character samples)
                return random.choice(self.arabic_words)
            else:
                # Phrase-level sample (30% of non-character samples)
                num_words = random.randint(2, self.max_phrase_length)
                words = random.sample(self.arabic_words, min(num_words, len(self.arabic_words)))
                return " ".join(words)
        
    def _find_arabic_fonts(self):
        """Try to find Arabic fonts on the system."""
        # Common Arabic font paths
        common_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "C:/Windows/Fonts/tahoma.ttf",  # Windows (has Arabic support)
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                self.font_paths.append(path)
                break
        
        if not self.font_paths:
            # Fallback: use default PIL font (may not support Arabic well)
            self.font_paths = [None]
    
    def _render_arabic_text(
        self,
        text: str,
        font_path: Optional[str],
        font_size: int,
        text_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int],
    ) -> Image.Image:
        """Render Arabic text on a background image."""
        # Reshape Arabic text for proper display
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        
        # Create image
        img = Image.new('RGB', (self.size, self.size), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Load font
        if font_path and os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), bidi_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (self.size - text_width) // 2
        y = (self.size - text_height) // 2
        
        # Draw text
        draw.text((x, y), bidi_text, fill=text_color, font=font)
        
        return img
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Compose Arabic text (character, word, or phrase based on mode)
        arabic_text = self._compose_text()
        
        # Randomly select font, colors, size
        font_path = random.choice(self.font_paths) if self.font_paths else None
        font_size = random.choice(self.font_sizes)
        bg_color = random.choice(self.background_colors)
        text_color = random.choice(self.text_colors)
        
        # Render image
        image = self._render_arabic_text(arabic_text, font_path, font_size, text_color, bg_color)
        
        # Convert to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        pixel_values = transform(image)
        
        # Generate prompt (English description or Arabic text)
        if self.use_english_prompts:
            # Production mode: Use English description
            prompt_template = random.choice(self.prompt_templates)
            prompt = prompt_template.format(text=arabic_text)
        else:
            # Research mode: Use just Arabic text
            prompt = arabic_text
        
        return {
            "pixel_values": pixel_values,
            "prompt": prompt,  # English description or Arabic text
            "text": arabic_text,   # Ground truth Arabic text for OCR evaluation
        }


class EvArESTDataset(Dataset):
    """
    EvArEST dataset loader.
    
    This is the SECONDARY dataset, used ONLY for:
    - OCR-based evaluation
    - OCR-guided loss
    - Weak supervision
    
    DO NOT treat EvArEST as a text-to-image dataset.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train" or "test"
        size: int = 1024,
        recognition_only: bool = True,  # Only use recognition dataset
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.size = size
        self.recognition_only = recognition_only
        
        if recognition_only:
            # Load recognition dataset (cropped word images)
            self.images_dir = self.data_dir / "Recognition" / split
            self.gt_file = self.data_dir / "Recognition" / f"{split}_gt.txt"
            
            if not self.images_dir.exists():
                raise ValueError(f"Recognition images directory not found: {self.images_dir}")
            if not self.gt_file.exists():
                raise ValueError(f"Ground truth file not found: {self.gt_file}")
            
            # Load ground truth
            self.samples = []
            with open(self.gt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        img_name, text = parts
                        img_path = self.images_dir / img_name
                        if img_path.exists():
                            self.samples.append({
                                "image_path": str(img_path),
                                "text": text,
                            })
        else:
            # Detection dataset (full images with annotations)
            # This is more complex and not recommended for training
            raise NotImplementedError("Detection dataset not implemented. Use recognition_only=True.")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Resize
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        pixel_values = transform(image)
        
        return {
            "pixel_values": pixel_values,
            "prompt": sample["text"],  # Ground truth text
            "text": sample["text"],    # For OCR evaluation
            "is_evarest": True,        # Flag to indicate this is from EvArEST
        }


class CombinedArabicDataset(Dataset):
    """
    Combines synthetic and EvArEST datasets.
    
    Synthetic data is primary, EvArEST is used for validation/weak supervision.
    """
    
    def __init__(
        self,
        synthetic_dataset: SyntheticArabicDataset,
        evarest_dataset: Optional[EvArESTDataset] = None,
        evarest_weight: float = 0.1,  # Weight of EvArEST samples
    ):
        self.synthetic_dataset = synthetic_dataset
        self.evarest_dataset = evarest_dataset
        self.evarest_weight = evarest_weight
        
        self.synthetic_len = len(synthetic_dataset)
        self.evarest_len = len(evarest_dataset) if evarest_dataset else 0
        
        # Calculate effective lengths
        if evarest_dataset:
            # Scale EvArEST to match weight
            self.evarest_effective_len = int(self.synthetic_len * evarest_weight)
            self.total_len = self.synthetic_len + self.evarest_effective_len
        else:
            self.evarest_effective_len = 0
            self.total_len = self.synthetic_len
    
    def __len__(self) -> int:
        return self.total_len
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.synthetic_len:
            return self.synthetic_dataset[idx]
        else:
            # Sample from EvArEST
            evarest_idx = (idx - self.synthetic_len) % self.evarest_len
            return self.evarest_dataset[evarest_idx]

