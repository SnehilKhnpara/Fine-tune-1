"""
Hybrid generation utilities: Base model for backgrounds, LoRA for text.
"""

import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Optional

# Try to import cv2, fallback to PIL-only methods if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Some text extraction methods may not work.")


def extract_text_region(
    text_image: Image.Image,
    background_image: Image.Image,
    method: str = "difference"
) -> Tuple[Image.Image, Image.Image]:
    """
    Extract text region from LoRA-generated image.
    
    Args:
        text_image: Image generated with LoRA (text on simple background)
        background_image: Image generated with base model (full scene)
        method: Extraction method ("difference", "threshold", "edge")
    
    Returns:
        Tuple of (text_mask, text_only_image)
    """
    if not CV2_AVAILABLE:
        # Fallback to simple PIL-based method
        return simple_text_extraction(text_image)
    
    # Ensure same size
    if text_image.size != background_image.size:
        text_image = text_image.resize(background_image.size, Image.LANCZOS)
    
    # Convert to numpy arrays
    text_np = np.array(text_image.convert("RGB"))
    bg_np = np.array(background_image.convert("RGB"))
    
    if method == "difference":
        # Method 1: Difference-based (works when backgrounds are similar)
        diff = cv2.absdiff(text_np, bg_np)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        # Threshold to get text regions
        _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
    elif method == "threshold":
        # Method 2: Threshold on text image (assumes simple background)
        gray = cv2.cvtColor(text_np, cv2.COLOR_RGB2GRAY)
        
        # Assume text is darker than background (adjust if needed)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if text is lighter
        if np.mean(gray[mask > 0]) > np.mean(gray[mask == 0]):
            mask = 255 - mask
            
    elif method == "edge":
        # Method 3: Edge detection
        gray = cv2.cvtColor(text_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect text regions
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Convert to PIL
    mask_pil = Image.fromarray(mask).convert("L")
    
    # Extract text only
    text_only = Image.new("RGB", text_image.size, (255, 255, 255))
    text_only.paste(text_image, mask=mask_pil)
    
    return mask_pil, text_only


def composite_text_on_background(
    background: Image.Image,
    text_image: Image.Image,
    text_mask: Optional[Image.Image] = None,
    blend_mode: str = "normal",
    opacity: float = 1.0
) -> Image.Image:
    """
    Composite text onto background image.
    
    Args:
        background: Base model generated background
        text_image: LoRA-generated text image (text only, white background)
        text_mask: Optional mask for text region
        blend_mode: "normal", "multiply", "overlay"
        opacity: Text opacity (0.0 to 1.0)
    
    Returns:
        Composited image
    """
    # Ensure same size
    if text_image.size != background.size:
        text_image = text_image.resize(background.size, Image.LANCZOS)
    
    if text_mask is None:
        # Create mask from text image (assume text is non-white)
        text_np = np.array(text_image.convert("RGB"))
        if CV2_AVAILABLE:
            gray = cv2.cvtColor(text_np, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        else:
            # PIL-only: simple threshold
            gray = np.array(text_image.convert("L"))
            mask = np.where(gray < 250, 255, 0).astype(np.uint8)
        text_mask = Image.fromarray(mask).convert("L")
    
    # Resize mask if needed
    if text_mask.size != background.size:
        text_mask = text_mask.resize(background.size, Image.LANCZOS)
    
    # Apply opacity to mask
    if opacity < 1.0:
        enhancer = ImageEnhance.Brightness(text_mask)
        text_mask = enhancer.enhance(opacity)
    
    # Composite
    if blend_mode == "normal":
        result = Image.composite(text_image, background, text_mask)
    elif blend_mode == "multiply":
        # Darken mode - paste text directly
        result = background.copy()
        result.paste(text_image, (0, 0), text_mask)
    elif blend_mode == "overlay":
        # Overlay mode - blend then composite
        blended = Image.blend(background, text_image, 0.3)
        result = Image.composite(text_image, blended, text_mask)
    else:
        result = Image.composite(text_image, background, text_mask)
    
    return result


def simple_text_extraction(text_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    Simple text extraction assuming white/light background.
    Extracts dark text regions.
    Works with or without cv2.
    """
    img_np = np.array(text_image.convert("RGB"))
    
    if CV2_AVAILABLE:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Adaptive threshold for text extraction
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    else:
        # PIL-only fallback: simple threshold
        gray = np.array(text_image.convert("L"))
        # Assume text is darker than background (threshold at 200)
        mask = np.where(gray < 200, 255, 0).astype(np.uint8)
    
    # Extract text
    text_only = Image.new("RGB", text_image.size, (255, 255, 255))
    mask_pil = Image.fromarray(mask).convert("L")
    text_only.paste(text_image, mask=mask_pil)
    
    return text_only, mask_pil
