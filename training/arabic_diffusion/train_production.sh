#!/bin/bash
# Production-Ready Arabic LoRA Training Script
# This script trains a LoRA that can generate ANY Arabic text, not just trained words

# Configuration
ARABIC_WORDS_FILE="Output/arabic_words.txt"  # 11,517 words
OUTPUT_DIR="arabic-lora-production"
PRETRAINED_MODEL="stabilityai/stable-diffusion-3-medium-diffusers"

# Training parameters optimized for production
python train_lora.py \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --arabic_words_file "$ARABIC_WORDS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --gradient_checkpointing \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4 \
    --resolution 512 \
    --synthetic_num_samples 500000 \
    --max_train_steps 100000 \
    --checkpointing_steps 5000 \
    --checkpoints_total_limit 10 \
    --enable_ocr_loss \
    --ocr_loss_weight 0.2 \
    --enable_rtl_loss \
    --rtl_loss_weight 0.1 \
    --use_english_prompts \
    --composition_mode mixed \
    --max_phrase_length 5 \
    --character_level_prob 0.1 \
    --rank 8 \
    --train_text_encoder

echo "Production training started. This will take several days."
echo "The model will learn to generate ANY Arabic text, not just trained words."

