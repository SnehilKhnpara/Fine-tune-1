"""
OCR integration for Arabic text recognition during training.

Supports:
- PaddleOCR (preferred for Arabic)
- Tesseract (fallback)

NOTE: OCR is ONLY used during training, NEVER during inference.
"""

import os
from typing import Optional, Union
import torch
from PIL import Image


class OCRWrapper:
    """
    Wrapper for OCR models.
    
    Supports PaddleOCR and Tesseract.
    """
    
    def __init__(self, ocr_type: str = "paddleocr", lang: str = "ar"):
        """
        Initialize OCR wrapper.
        
        Args:
            ocr_type: "paddleocr" or "tesseract"
            lang: Language code (default: "ar" for Arabic)
        """
        self.ocr_type = ocr_type.lower()
        self.lang = lang
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the OCR model."""
        if self.ocr_type == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                # Newer versions of PaddleOCR no longer accept `use_gpu` as an argument.
                # To stay compatible across versions, we avoid passing it explicitly
                # and let PaddleOCR decide based on the installed paddlepaddle backend.
                self.model = PaddleOCR(use_angle_cls=True, lang='ar')
                print("PaddleOCR initialized successfully")
            except ImportError:
                print("PaddleOCR not available, falling back to Tesseract")
                self.ocr_type = "tesseract"
                self._initialize()
            except Exception as e:
                print(f"Error initializing PaddleOCR: {e}, falling back to Tesseract")
                self.ocr_type = "tesseract"
                self._initialize()
        
        if self.ocr_type == "tesseract":
            try:
                import pytesseract
                # Check if Arabic language data is available
                try:
                    pytesseract.get_languages()
                except:
                    pass
                self.model = pytesseract
                print("Tesseract initialized successfully")
            except ImportError:
                print("Tesseract not available. OCR loss will be disabled.")
                self.model = None
            except Exception as e:
                print(f"Error initializing Tesseract: {e}. OCR loss will be disabled.")
                self.model = None
    
    def ocr(self, image: Union[Image.Image, torch.Tensor]) -> str:
        """
        Run OCR on image.
        
        Args:
            image: PIL Image or torch Tensor (C, H, W) in range [-1, 1]
        
        Returns:
            Extracted text string
        """
        if self.model is None:
            return ""
        
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            from torchvision import transforms
            # Denormalize
            image = (image + 1.0) / 2.0
            image = torch.clamp(image, 0, 1)
            to_pil = transforms.ToPILImage()
            image = to_pil(image.cpu())
        
        try:
            if self.ocr_type == "paddleocr":
                # PaddleOCR format. Newer versions may not accept `cls` arg,
                # so we try with it first, then fall back without it.
                try:
                    result = self.model.ocr(image, cls=False)
                except TypeError:
                    # Older / different API: no `cls` keyword
                    result = self.model.ocr(image)
                if result and result[0] and len(result[0]) > 0:
                    # Extract text from first detection
                    text = result[0][0][1][0]  # Format: [[bbox, (text, confidence)]]
                    return text
                return ""
            
            elif self.ocr_type == "tesseract":
                # Tesseract format
                lang_code = "ara" if self.lang == "ar" else self.lang
                text = self.model.image_to_string(image, lang=lang_code)
                return text.strip()
        
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
        
        return ""
    
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self.model is not None

