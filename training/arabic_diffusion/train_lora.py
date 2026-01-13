#!/usr/bin/env python
# coding=utf-8
"""
Training script for Arabic diffusion fine-tuning using LoRA.

This script implements Approach 3: Fine-tuned Arabic Diffusion.
It trains LoRA adapters to improve Arabic text understanding and rendering.

IMPORTANT:
- This is a RESEARCH phase, not production dependency
- Production correctness MUST remain guaranteed by Approach 1 (mask/overlay)
- OCR is ONLY used during training, NEVER during inference
"""

import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    FluxTransformer2DModel,
    FluxPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module

# Import our custom modules
try:
    from .dataset import SyntheticArabicDataset, EvArESTDataset, CombinedArabicDataset
    from .losses import CombinedLoss, OCRGuidedLoss, RTLDirectionalityLoss
    from .ocr_loss import OCRWrapper
except ImportError:
    # Fallback for direct execution
    from dataset import SyntheticArabicDataset, EvArESTDataset, CombinedArabicDataset
    from losses import CombinedLoss, OCRGuidedLoss, RTLDirectionalityLoss
    from ocr_loss import OCRWrapper

if is_wandb_available():
    import wandb

check_min_version("0.30.0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Arabic diffusion LoRA training script.")
    
    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="sd3",
        choices=["sd3", "flux"],
        help="Model type to fine-tune: 'sd3' for Stable Diffusion 3 or 'flux' for FLUX.1-dev",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models. If None, uses default based on model_type.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files (e.g., fp16).",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--arabic_words_file",
        type=str,
        default=None,
        help="Path to file containing Arabic words (one per line).",
    )
    parser.add_argument(
        "--evarest_data_dir",
        type=str,
        default=None,
        help="Path to EvArEST dataset directory.",
    )
    parser.add_argument(
        "--synthetic_num_samples",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate.",
    )
    parser.add_argument(
        "--evarest_weight",
        type=float,
        default=0.1,
        help="Weight of EvArEST samples in combined dataset.",
    )
    parser.add_argument(
        "--use_english_prompts",
        action="store_true",
        default=True,
        help="Use English descriptions in prompts (production mode). Default: True",
    )
    parser.add_argument(
        "--composition_mode",
        type=str,
        default="mixed",
        choices=["word", "phrase", "character", "mixed"],
        help="Text composition mode: 'word' (single words), 'phrase' (multiple words), 'character' (single chars), 'mixed' (all). Default: 'mixed' (production mode)",
    )
    parser.add_argument(
        "--max_phrase_length",
        type=int,
        default=5,
        help="Maximum number of words in a phrase. Default: 5",
    )
    parser.add_argument(
        "--character_level_prob",
        type=float,
        default=0.1,
        help="Probability of character-level samples in mixed mode. Default: 0.1",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arabic-lora-output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices. Must be between 4 and 32 (inclusive).",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder.",
    )
    
    # OCR and loss arguments
    parser.add_argument(
        "--enable_ocr_loss",
        action="store_true",
        help="Enable OCR-guided loss.",
    )
    parser.add_argument(
        "--ocr_type",
        type=str,
        default="paddleocr",
        choices=["paddleocr", "tesseract"],
        help="OCR engine to use.",
    )
    parser.add_argument(
        "--ocr_loss_weight",
        type=float,
        default=0.1,
        help="Weight for OCR-guided loss.",
    )
    parser.add_argument(
        "--enable_rtl_loss",
        action="store_true",
        help="Enable RTL directionality penalty.",
    )
    parser.add_argument(
        "--rtl_loss_weight",
        type=float,
        default=0.05,
        help="Weight for RTL directionality loss.",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="The integration to report results and logs to.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    
    # SD3-specific arguments
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help="Maximum sequence length for T5 text encoder.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs.",
    )
    # Showcase samples are enabled by default
    parser.add_argument(
        "--no_save_showcase_samples",
        action="store_false",
        dest="save_showcase_samples",
        default=True,
        help="Disable saving showcase samples of first N words (enabled by default).",
    )
    parser.add_argument(
        "--num_showcase_words",
        type=int,
        default=5,
        help="Number of words to generate showcase samples for. Default: 5",
    )
    parser.add_argument(
        "--variations_per_word",
        type=int,
        default=8,
        help="Number of variations (different sizes/fonts/colors) per word. Default: 8",
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.arabic_words_file is None:
        raise ValueError("--arabic_words_file is required. Provide a file with Arabic words (one per line).")
    
    if not os.path.exists(args.arabic_words_file):
        raise ValueError(f"Arabic words file not found: {args.arabic_words_file}")
    
    return args


def load_arabic_words(file_path: str) -> List[str]:
    """Load Arabic words from file (one per line, or word frequency format)."""
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle word frequency format: "word 123456" -> extract just "word"
            # Split by whitespace and take the first part (the word)
            word = line.split()[0] if line.split() else line
            if word:
                words.append(word)
    return words


def main():
    args = parse_args()
    
    # Validate rank parameter
    if args.rank < 4 or args.rank > 32:
        raise ValueError(f"--rank must be between 4 and 32 (inclusive), got {args.rank}")
    
    logging_dir = Path(args.output_dir, "logs")
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging.")
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Arabic words
    arabic_words = load_arabic_words(args.arabic_words_file)
    logger.info(f"Loaded {len(arabic_words)} Arabic words")
    
    # Generate showcase images for first 5 words (for client demo)
    if accelerator.is_main_process and args.save_showcase_samples:
        logger.info("Generating showcase samples for first 5 words...")
        showcase_dir = os.path.join(args.output_dir, "showcase_samples")
        os.makedirs(showcase_dir, exist_ok=True)
        
        # Get first N words (default 5)
        num_showcase = min(args.num_showcase_words, len(arabic_words))
        showcase_words = arabic_words[:num_showcase]
        
        logger.info(f"  Generating showcase for {num_showcase} words: {showcase_words}")
        
        # Generate variations for each word - use same settings as dataset
        # Import here since they're needed for showcase generation
        import arabic_reshaper
        from bidi.algorithm import get_display
        
        # Use dataset's render method via a temporary dataset instance
        temp_dataset = SyntheticArabicDataset(
            arabic_words=showcase_words,
            size=args.resolution,
            num_samples=1,  # Not used, just for initialization
            use_english_prompts=False,  # Don't need prompts for showcase
            composition_mode="word",
        )
        
        font_sizes = temp_dataset.font_sizes if hasattr(temp_dataset, 'font_sizes') else [32, 40, 48, 56, 64]
        background_colors = temp_dataset.background_colors if hasattr(temp_dataset, 'background_colors') else [(255, 255, 255), (240, 240, 240), (250, 250, 250)]
        text_colors = temp_dataset.text_colors if hasattr(temp_dataset, 'text_colors') else [(0, 0, 0), (50, 50, 50), (20, 20, 20)]
        font_paths = temp_dataset.font_paths if hasattr(temp_dataset, 'font_paths') and temp_dataset.font_paths else [None]
        
        sample_count = 0
        variations_per_word = args.variations_per_word
        
        for word_idx, word in enumerate(showcase_words):
            # Create folder for this word (sanitize word for filename)
            safe_word = "".join(c for c in word if c.isalnum() or c in (' ', '-', '_')).strip()[:20]  # Limit length
            word_showcase_dir = os.path.join(showcase_dir, f"word_{word_idx+1}_{safe_word}")
            os.makedirs(word_showcase_dir, exist_ok=True)
            
            logger.info(f"    Word {word_idx+1}/{num_showcase}: {word} - Generating {variations_per_word} variations...")
            
            # Generate multiple variations for each word
            for var_idx in range(variations_per_word):
                # Randomly select font, size, colors
                font_path = random.choice(font_paths) if font_paths else None
                font_size = random.choice(font_sizes)
                bg_color = random.choice(background_colors)
                text_color = random.choice(text_colors)
                
                # Use dataset's render method
                try:
                    img = temp_dataset._render_arabic_text(word, font_path, font_size, text_color, bg_color)
                    
                    # Save image with descriptive filename
                    font_name = os.path.basename(font_path) if font_path else "default"
                    # Sanitize font name for filename (remove extension, limit length)
                    font_name_clean = "".join(c for c in os.path.splitext(font_name)[0] if c.isalnum() or c in ('-', '_'))[:12]
                    img_filename = f"var_{var_idx+1:02d}_size{font_size}_font{font_name_clean}_bg{'_'.join(map(str, bg_color))}_text{'_'.join(map(str, text_color))}.png"
                    # Sanitize full filename (remove any problematic chars, keep ASCII safe chars)
                    img_filename = "".join(c if (c.isalnum() or c in ('-', '_', '.')) else '_' for c in img_filename)
                    img_path = os.path.join(word_showcase_dir, img_filename)
                    img.save(img_path)
                    sample_count += 1
                except Exception as e:
                    logger.warning(f"      Failed to generate variation {var_idx+1} for word '{word}': {e}")
                    continue
        
        logger.info(f"✓ Generated {sample_count} showcase samples in {showcase_dir}")
        logger.info(f"  Showcase folder: {showcase_dir}")
        logger.info(f"  {num_showcase} words × {variations_per_word} variations each = {sample_count} total samples")
        logger.info("  Continuing with training...\n")
    
    # Create datasets
    synthetic_dataset = SyntheticArabicDataset(
        arabic_words=arabic_words,
        size=args.resolution,
        num_samples=args.synthetic_num_samples,
        use_english_prompts=args.use_english_prompts,
        composition_mode=args.composition_mode,
        max_phrase_length=args.max_phrase_length,
        character_level_prob=args.character_level_prob,
    )
    
    evarest_dataset = None
    if args.evarest_data_dir:
        try:
            evarest_dataset = EvArESTDataset(
                data_dir=args.evarest_data_dir,
                split="train",
                size=args.resolution,
                recognition_only=True,
            )
            logger.info(f"Loaded EvArEST dataset with {len(evarest_dataset)} samples")
        except Exception as e:
            logger.warning(f"Could not load EvArEST dataset: {e}. Continuing without it.")
    
    train_dataset = CombinedArabicDataset(
        synthetic_dataset=synthetic_dataset,
        evarest_dataset=evarest_dataset,
        evarest_weight=args.evarest_weight,
    )
    
    # Initialize OCR (if enabled)
    ocr_wrapper = None
    if args.enable_ocr_loss:
        ocr_wrapper = OCRWrapper(ocr_type=args.ocr_type, lang="ar")
        if not ocr_wrapper.is_available():
            logger.warning("OCR is not available. OCR loss will be disabled.")
            args.enable_ocr_loss = False
    
    # Create loss functions
    ocr_loss_fn = None
    if args.enable_ocr_loss and ocr_wrapper:
        ocr_loss_fn = OCRGuidedLoss(
            ocr_model=ocr_wrapper,
            weight=args.ocr_loss_weight,
        )
    
    rtl_loss_fn = None
    if args.enable_rtl_loss:
        rtl_loss_fn = RTLDirectionalityLoss(weight=args.rtl_loss_weight)
    
    combined_loss_fn = CombinedLoss(
        ocr_loss_fn=ocr_loss_fn,
        rtl_loss_fn=rtl_loss_fn,
        ocr_loss_weight=args.ocr_loss_weight,
        rtl_loss_weight=args.rtl_loss_weight,
    )
    
    # Set default model path based on model_type if not provided
    if args.pretrained_model_name_or_path is None:
        if args.model_type == "sd3":
            args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"
        elif args.model_type == "flux":
            args.pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
        else:
            raise ValueError(f"Unknown model_type: {args.model_type}")
    
    logger.info(f"Using model_type: {args.model_type}")
    logger.info(f"Loading model from: {args.pretrained_model_name_or_path}")
    
    # Load models based on model_type
    if args.model_type == "sd3":
        # SD3: 3 text encoders (CLIP + CLIP + T5)
        tokenizer_one = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
        )
        tokenizer_three = T5TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_3",
            revision=args.revision,
        )
        
        # Import text encoder classes
        text_encoder_config = PretrainedConfig.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        model_class = text_encoder_config.architectures[0]
        if model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection
            text_encoder_cls_one = CLIPTextModelWithProjection
            text_encoder_cls_two = CLIPTextModelWithProjection
        else:
            raise ValueError(f"{model_class} is not supported.")
        
        from transformers import T5EncoderModel
        text_encoder_cls_three = T5EncoderModel
        
        # Load models
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
        )
        text_encoder_three = text_encoder_cls_three.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
        )
        
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
        
        # Freeze models
        transformer.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_three.requires_grad_(False)
        
        # Store for later use
        text_encoder_three_exists = True
        
    elif args.model_type == "flux":
        # Flux: 2 text encoders (CLIP + T5)
        from transformers import CLIPTextModel, CLIPTokenizer
        
        tokenizer_one = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
        tokenizer_two = T5TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
        )
        tokenizer_three = None  # Flux doesn't have a third tokenizer
        
        # Import text encoder classes
        from transformers import T5EncoderModel
        
        # Load models
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        
        text_encoder_one = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        text_encoder_two = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
        )
        text_encoder_three = None  # Flux doesn't have a third text encoder
        
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
        
        # Freeze models
        transformer.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        
        # Store for later use
        text_encoder_three_exists = False
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    
    # Set weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    if text_encoder_three_exists:
        text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
    
    # Add LoRA adapters
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)
    
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)
    
    # Create data loader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # CRITICAL: Cast trainable parameters to FP32 for mixed precision training
    # This ensures gradients are in FP32, which the scaler can handle properly
    if accelerator.mixed_precision in ["fp16", "bf16"]:
        models_to_cast = [transformer]
        if args.train_text_encoder:
            models_to_cast.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models_to_cast, dtype=torch.float32)
    
    # Optimizer
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
    
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    if args.train_text_encoder:
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            transformer_parameters_with_lr,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]
    else:
        params_to_optimize = [transformer_parameters_with_lr]
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )
    
    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # CRITICAL: Cast trainable parameters to FP32 for mixed precision training
    # This ensures gradients are in FP32, which the scaler can handle properly
    # Must be done BEFORE accelerator.prepare()
    if accelerator.mixed_precision in ["fp16", "bf16"]:
        models_to_cast = [transformer]
        if args.train_text_encoder:
            models_to_cast.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models_to_cast, dtype=torch.float32)
    
    # Prepare with accelerator
    if args.train_text_encoder:
        transformer, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
    
    # Initialize trackers
    if accelerator.is_main_process:
        tracker_name = "arabic-diffusion-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))
    
    # Training loop
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new training.")
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # Encode prompt helper - adapted from train_dreambooth_lora_sd3.py
    def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        max_sequence_length,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        
        return prompt_embeds
    
    def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
        
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        
        return prompt_embeds, pooled_prompt_embeds
    
    def encode_prompt(
        text_encoders,
        tokenizers,
        prompt: str,
        max_sequence_length,
        device=None,
        num_images_per_prompt: int = 1,
        model_type="sd3",
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        
        if model_type == "sd3":
            # SD3: 2 CLIP encoders + 1 T5 encoder
            clip_tokenizers = tokenizers[:2]
            clip_text_encoders = text_encoders[:2]
            
            clip_prompt_embeds_list = []
            clip_pooled_prompt_embeds_list = []
            for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
                prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device if device is not None else text_encoder.device,
                    num_images_per_prompt=num_images_per_prompt,
                )
                clip_prompt_embeds_list.append(prompt_embeds)
                clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            
            clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)
            
            t5_prompt_embed = _encode_prompt_with_t5(
                text_encoders[-1],
                tokenizers[-1],
                max_sequence_length,
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device if device is not None else text_encoders[-1].device,
            )
            
            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )
            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            
        elif model_type == "flux":
            # Flux: 1 CLIP encoder + 1 T5 encoder
            clip_prompt_embeds, clip_pooled_prompt_embeds = _encode_prompt_with_clip(
                text_encoder=text_encoders[0],
                tokenizer=tokenizers[0],
                prompt=prompt,
                device=device if device is not None else text_encoders[0].device,
                num_images_per_prompt=num_images_per_prompt,
            )
            
            t5_prompt_embed = _encode_prompt_with_t5(
                text_encoders[1],
                tokenizers[1],
                max_sequence_length,
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device if device is not None else text_encoders[1].device,
            )
            
            # Flux concatenates CLIP and T5 embeddings differently
            # Pad CLIP to match T5 dimension if needed
            if clip_prompt_embeds.shape[-1] != t5_prompt_embed.shape[-1]:
                clip_prompt_embeds = torch.nn.functional.pad(
                    clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
                )
            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = clip_pooled_prompt_embeds
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        return prompt_embeds, pooled_prompt_embeds
    
    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                prompts = batch["prompt"]  # List of Arabic text prompts
                target_texts = batch["text"]  # For OCR loss
                
                # Encode prompts properly using text encoders based on model_type
                if args.model_type == "sd3":
                    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
                    tokenizers_list = [tokenizer_one, tokenizer_two, tokenizer_three]
                elif args.model_type == "flux":
                    text_encoders = [text_encoder_one, text_encoder_two]
                    tokenizers_list = [tokenizer_one, tokenizer_two]
                else:
                    raise ValueError(f"Unsupported model_type: {args.model_type}")
                
                # Encode all prompts in the batch
                prompt_embeds_list = []
                pooled_prompt_embeds_list = []
                for prompt in prompts:
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=text_encoders,
                        tokenizers=tokenizers_list,
                        prompt=prompt,
                        max_sequence_length=args.max_sequence_length,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        model_type=args.model_type,
                    )
                    prompt_embeds_list.append(prompt_embeds)
                    pooled_prompt_embeds_list.append(pooled_prompt_embeds)
                
                # Stack into batch
                prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
                pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)
                
                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)
                
                # Sample noise
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                
                # Sample timesteps
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                
                # Add noise
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                
                # Ensure prompt embeddings are on correct device and dtype
                prompt_embeds = prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)
                
                # Predict
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                
                # Preconditioning
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input
                
                # Compute weighting
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                
                # Compute target
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input
                
                # Standard diffusion loss
                diffusion_loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                diffusion_loss = diffusion_loss.mean()

                # ------------------------------------------------------------------
                # OCR-guided & RTL losses (optional, controlled by CLI flags)
                # ------------------------------------------------------------------
                if args.enable_ocr_loss:
                    # Decode current model input latents to an RGB image for OCR.
                    # We use the *clean* latents (model_input) rather than noisy ones.
                    with torch.no_grad():
                        latents_for_ocr = model_input.detach().to(vae.dtype)
                        decoded = vae.decode(latents_for_ocr / vae.config.scaling_factor).sample  # (B, 3, H, W)
                        generated_image = decoded.clamp(-1, 1)
                else:
                    generated_image = None

                # Only pass prompt embeddings to RTL loss if it's enabled
                prompt_embeds_for_rtl = prompt_embeds if args.enable_rtl_loss else None

                # Combined loss: diffusion + (optional) OCR + (optional) RTL
                loss_dict = combined_loss_fn(
                    diffusion_loss=diffusion_loss,
                    generated_image=generated_image,
                    target_text=target_texts[0] if target_texts else None,
                    prompt_embeds=prompt_embeds_for_rtl,
                )
                loss = loss_dict["total_loss"]
                
                accelerator.backward(loss)
                
                # Check if using mixed precision (handle both string and enum types)
                is_fp16 = accelerator.mixed_precision == "fp16" or str(accelerator.mixed_precision) == "fp16"
                is_bf16 = accelerator.mixed_precision == "bf16" or str(accelerator.mixed_precision) == "bf16"
                use_mixed_precision = is_fp16 or is_bf16
                has_scaler = accelerator.scaler is not None
                
                if accelerator.sync_gradients:
                    # Collect parameters for gradient clipping
                    if args.train_text_encoder:
                        params_to_clip = list(itertools.chain(
                            transformer_lora_parameters, 
                            text_lora_parameters_one, 
                            text_lora_parameters_two
                        ))
                    else:
                        params_to_clip = list(transformer_lora_parameters)
                    
                    # Clip gradients - skip for FP16 to avoid scaler conflicts
                    # Gradient clipping is helpful but not critical for training
                    if not use_mixed_precision:
                        # For FP32: use accelerator's method (safe)
                        accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    # For FP16/BF16: skip gradient clipping to avoid scaler issues
                    # The scaler will handle gradient scaling automatically
                
                # Optimizer step
                # With cast_training_params, trainable params are in FP32
                # Use optimizer.step() directly - accelerator handles scaler internally
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                
                logs = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break
    
    # Save final LoRA weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        transformer = transformer.to(torch.float32)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        
        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            text_encoder_two = accelerator.unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = get_peft_model_state_dict(text_encoder_two.to(torch.float32))
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None
        
        # Save LoRA weights using appropriate pipeline based on model_type
        if args.model_type == "sd3":
            StableDiffusion3Pipeline.save_lora_weights(
                save_directory=args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers,
            )
        elif args.model_type == "flux":
            # For Flux, we only have 2 text encoders
            FluxPipeline.save_lora_weights(
                save_directory=args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers,
            )
        else:
            raise ValueError(f"Unsupported model_type: {args.model_type}")
        
        logger.info(f"Saved LoRA weights to {args.output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()

