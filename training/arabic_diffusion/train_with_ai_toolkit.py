#!/usr/bin/env python
# coding=utf-8
"""
Integration script for training with Ostris AI Toolkit.

This script prepares the dataset and generates a YAML config for AI Toolkit,
then optionally runs the training.
"""

import argparse
import os
import shutil
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
    Create AI Toolkit YAML config. AI Toolkit expects: job, config (with name + process), meta.
    See: https://github.com/ostris/ai-toolkit config/examples/train_lora_flux_24gb.yaml
    """
    if args.pretrained_model_name_or_path:
        model_path = args.pretrained_model_name_or_path
    elif args.model_type == "flux":
        model_path = "black-forest-labs/FLUX.1-dev"
    elif args.model_type == "sd3":
        model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    else:
        model_path = "black-forest-labs/FLUX.1-dev"

    if args.synthetic_num_samples and args.train_batch_size:
        total_steps = (args.synthetic_num_samples // args.train_batch_size) * args.num_train_epochs
    else:
        total_steps = args.max_train_steps or 3000

    dataset_path = os.path.abspath(dataset_dir)
    out_folder = os.path.abspath(args.output_dir)
    res = args.resolution or 1024
    is_flux = args.model_type == "flux"

    # Match AI Toolkit's expected layout: job, config (name + process), meta
    config = {
        "job": "extension",
        "config": {
            "name": f"arabic_{args.model_type}_lora",
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": out_folder,
                    "device": "cuda:0",
                    "network": {"type": "lora", "linear": args.rank, "linear_alpha": args.rank},
                    "save": {
                        "dtype": "float16",
                        "save_every": 250,
                        "max_step_saves_to_keep": 4,
                        "push_to_hub": False,
                    },
                    "datasets": [
                        {
                            "folder_path": dataset_path,
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [res] if res <= 768 else [768, 1024],
                        }
                    ],
                    "train": {
                        "batch_size": args.train_batch_size,
                        "steps": total_steps,
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": args.learning_rate,
                        "dtype": "bf16" if (is_flux or (getattr(args, "mixed_precision", None) == "bf16")) else "fp16",
                        "ema_config": {"use_ema": True, "ema_decay": 0.99},
                    },
                    "model": {
                        "name_or_path": model_path,
                        "is_flux": is_flux,
                        "quantize": True,
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": 250,
                        "width": res,
                        "height": res,
                        "prompts": [
                            "Arabic text on a sign, clean background",
                            "A poster with Arabic writing, simple layout",
                            "Arabic calligraphy on paper, white background",
                        ],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 4,
                        "sample_steps": 20,
                    },
                }
            ],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }

    os.makedirs(os.path.dirname(config_output_path) or ".", exist_ok=True)
    with open(config_output_path, "w", encoding="utf-8") as f:
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
    # 1) Explicit path: env AI_TOOLKIT_PATH, e.g. D:\Test projects\Fine-tune-1\ai-toolkit
    env_path = os.environ.get("AI_TOOLKIT_PATH", "").strip()
    if env_path and os.path.exists(env_path) and os.path.exists(os.path.join(env_path, "run.py")):
        ai_toolkit_path = os.path.abspath(env_path)
    # 2) Else search common locations
    if not ai_toolkit_path:
        for path in ["ai-toolkit", "../ai-toolkit",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ai-toolkit")]:
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
    
    # Create config file in output_dir (for your records). Use abspath so it works regardless of cwd.
    config_path = os.path.abspath(os.path.join(args.output_dir, "ai_toolkit_config.yaml"))
    create_ai_toolkit_config(args, dataset_dir, config_path)
    
    # Copy into ai-toolkit dir. AI Toolkit run.py needs to find this file.
    config_in_aitoolkit = os.path.join(ai_toolkit_path, "ai_toolkit_job_config.yaml")
    shutil.copy2(config_path, config_in_aitoolkit)
    
    # AI Toolkit requires the config path as absolute (and paths without spaces work best)
    config_arg = os.path.abspath(config_in_aitoolkit)
    if not os.path.isfile(config_arg):
        raise FileNotFoundError(f"Config not found after copy: {config_arg}")
    
    # Run AI Toolkit
    print("\n" + "="*60)
    print("Starting AI Toolkit training...")
    print("="*60 + "\n")
    
    # Change to AI Toolkit directory (needed for its imports) and run with absolute config path
    original_cwd = os.getcwd()
    try:
        os.chdir(ai_toolkit_path)
        
        cmd = [sys.executable, "run.py", config_arg]
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
