#!/bin/bash
# Training script for large dataset (11,517 words) - Better generalization

python training/arabic_diffusion/train_lora.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
    --arabic_words_file training/arabic_diffusion/Output/arabic_words.txt \
    --output_dir arabic-lora-output \
    --resolution 1024 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --text_encoder_lr 2e-6 \
    --train_text_encoder \
    --rank 8 \
    --synthetic_num_samples 50000 \
    --composition_mode mixed \
    --use_english_prompts \
    --enable_ocr_loss \
    --ocr_type paddleocr \
    --ocr_loss_weight 0.3 \
    --enable_rtl_loss \
    --rtl_loss_weight 0.1 \
    --lr_scheduler cosine \
    --lr_warmup_steps 500 \
    --gradient_checkpointing \
    --mixed_precision fp16 \
    --report_to tensorboard \
    --checkpointing_steps 1000 \
    --checkpoints_total_limit 3
