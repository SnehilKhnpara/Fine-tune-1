@echo off
REM Production-Ready Arabic LoRA Training Script for Windows
REM This trains a model that can generate ANY Arabic text with English prompts

echo Starting production training...
echo This will train a model that understands English prompts and can generate any Arabic text.
echo Training will take several days (100k steps).
echo.

cd training\arabic_diffusion

python train_lora.py ^
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers ^
    --arabic_words_file Output/arabic_words.txt ^
    --output_dir arabic-lora-production ^
    --train_batch_size 1 ^
    --num_train_epochs 1 ^
    --learning_rate 5e-5 ^
    --gradient_checkpointing ^
    --mixed_precision fp16 ^
    --gradient_accumulation_steps 4 ^
    --resolution 512 ^
    --synthetic_num_samples 500000 ^
    --max_train_steps 100000 ^
    --checkpointing_steps 5000 ^
    --checkpoints_total_limit 10 ^
    --enable_ocr_loss ^
    --ocr_loss_weight 0.2 ^
    --enable_rtl_loss ^
    --rtl_loss_weight 0.1 ^
    --use_english_prompts ^
    --composition_mode mixed ^
    --max_phrase_length 5 ^
    --character_level_prob 0.1 ^
    --rank 8 ^
    --train_text_encoder

echo.
echo Training started. Check arabic-lora-production directory for checkpoints.
pause

