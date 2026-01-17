#!/usr/bin/env python
# coding=utf-8
"""
Integration script for training with Ostris AI Toolkit.

This script prepares the dataset and generates a YAML config for AI Toolkit,
then optionally runs the training.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required for AI Toolkit integration.")
    print("Install it with: pip install pyyaml")
    sys.exit(1)


def create_ai_toolkit_config(
    args,
    dataset_dir: str,
    config_output_path: str,
) -> str:
    """
    Create AI Toolkit YAML configuration file.
    
    Args:
        args: Parsed arguments from train_lora.py
        dataset_dir: Path to pre-generated dataset directory
        config_output_path: Where to save the config YAML
    
    Returns:
        Path to created config file
    """
    # Determine model path
    if args.pretrained_model_name_or_path:
        model_path = args.pretrained_model_name_or_path
    elif args.model_type == "flux":
        model_path = "black-forest-labs/FLUX.1-dev"
    elif args.model_type == "sd3":
        model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    else:
        model_path = "black-forest-labs/FLUX.1-dev"
    
    # Calculate steps from epochs
    # AI Toolkit uses steps, not epochs
    # Approximate: steps = (num_samples / batch_size) * epochs
    if args.synthetic_num_samples and args.train_batch_size:
        steps_per_epoch = args.synthetic_num_samples // args.train_batch_size
        total_steps = steps_per_epoch * args.num_train_epochs
    else:
        total_steps = args.max_train_steps or 3000
    
    # Create config structure
    config = {
        "job": {
            "name": f"arabic_{args.model_type}_lora",
            "training_folder": args.output_dir,
        },
        "process": [
            {
                "type": "flux_trainer" if args.model_type == "flux" else "sd3_trainer",
                "folder_path": os.path.abspath(dataset_dir),
                "steps": total_steps,
                "batch_size": args.train_batch_size,
                "learning_rate": args.learning_rate,
                "model": {
                    "name_or_path": model_path,
                    "is_flux": args.model_type == "flux",
                    "quantize": False,  # Set to True if you need low-VRAM mode
                },
                "lora": {
                    "rank": args.rank,
                    "alpha": args.rank,  # Common practice: alpha = rank
                },
                "optimizer": {
                    "type": "adamw",
                    "lr": args.learning_rate,
                },
                "scheduler": {
                    "type": args.lr_scheduler,
                    "warmup_steps": args.lr_warmup_steps,
                },
            }
        ],
    }
    
    # Add mixed precision if specified
    if args.mixed_precision in ["fp16", "bf16"]:
        config["process"][0]["mixed_precision"] = args.mixed_precision
    
    # Save config
    os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
    with open(config_output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created AI Toolkit config: {config_output_path}")
    return config_output_path


def check_ai_toolkit_installed() -> bool:
    """Check if AI Toolkit is installed."""
    try:
        # Try to import or check if ai-toolkit directory exists
        # AI Toolkit is typically installed as a git repo
        return os.path.exists("ai-toolkit") or os.path.exists("../ai-toolkit")
    except:
        return False


def train_with_ai_toolkit(args):
    """
    Main function to train with AI Toolkit.
    
    This function:
    1. Checks if dataset is pre-generated (if not, generates it)
    2. Creates AI Toolkit config YAML
    3. Runs AI Toolkit training
    """
    print("\n" + "="*60)
    print("Training with Ostris AI Toolkit")
    print("="*60 + "\n")
    
    # Check if AI Toolkit is installed
    ai_toolkit_path = None
    possible_paths = [
        "ai-toolkit",
        "../ai-toolkit",
        os.path.join(os.path.dirname(__file__), "..", "..", "ai-toolkit"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "run.py")):
            ai_toolkit_path = os.path.abspath(path)
            break
    
    if not ai_toolkit_path:
        print("ERROR: AI Toolkit not found!")
        print("\nPlease install AI Toolkit first:")
        print("  git clone https://github.com/ostris/ai-toolkit.git")
        print("  cd ai-toolkit")
        print("  pip install -r requirements.txt")
        print("\nOr place it in one of these locations:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        sys.exit(1)
    
    print(f"✓ Found AI Toolkit at: {ai_toolkit_path}\n")
    
    # Check if dataset is pre-generated
    dataset_dir = os.path.join(args.output_dir, "ai_toolkit_dataset")
    
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print("Dataset not found. Generating dataset for AI Toolkit...")
        print(f"Output directory: {dataset_dir}\n")
        
        # Import and run dataset generation
        try:
            from .prepare_ai_toolkit_dataset import generate_dataset
        except ImportError:
            from prepare_ai_toolkit_dataset import generate_dataset
        
        # Generate dataset using the helper function
        generate_dataset(
            arabic_words_file=args.arabic_words_file,
            output_dir=dataset_dir,
            num_samples=args.synthetic_num_samples,
            image_size=args.resolution,
            composition_mode=args.composition_mode,
            max_phrase_length=args.max_phrase_length,
            character_level_prob=args.character_level_prob,
            use_english_prompts=args.use_english_prompts,
        )
        
        print(f"✓ Dataset generated: {dataset_dir}\n")
    else:
        print(f"✓ Using existing dataset: {dataset_dir}\n")
    
    # Create config file
    config_path = os.path.join(args.output_dir, "ai_toolkit_config.yaml")
    create_ai_toolkit_config(args, dataset_dir, config_path)
    
    # Run AI Toolkit
    print("\n" + "="*60)
    print("Starting AI Toolkit training...")
    print("="*60 + "\n")
    
    # Change to AI Toolkit directory and run
    original_cwd = os.getcwd()
    try:
        os.chdir(ai_toolkit_path)
        
        # Run AI Toolkit
        cmd = [sys.executable, "run.py", "--config", config_path]
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {ai_toolkit_path}\n")
        
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            print(f"\nERROR: AI Toolkit training failed with exit code {result.returncode}")
            sys.exit(1)
        
        print("\n✓ Training completed successfully!")
        print(f"  Check output directory: {args.output_dir}")
        
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    # This can be called directly for testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file path")
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print("Config loaded:", config)
    else:
        print("This script is typically called from train_lora.py")
        print("Use: python train_lora.py --training_method ai_toolkit ...")
