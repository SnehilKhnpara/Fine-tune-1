@echo off
REM Quick Demo Training - Fast, Quality Solution for Client Demo
REM Trains on 183 words with English prompts - Ready in ~12-24 hours

echo ========================================
echo Quick Demo Training for Client
echo ========================================
echo.
echo This will train a model that:
echo - Works with English prompts
echo - Generates proper backgrounds
echo - Renders correct Arabic text
echo - Ready in 12-24 hours (not 5-7 days)
echo.
echo Training on: 183 words (from data/arabic_words.txt)
echo Training steps: 20,000 (fast but quality)
echo.
pause

cd training\arabic_diffusion

python train_lora.py ^
    --arabic_words_file ../../data/arabic_words.txt ^
    --output_dir arabic-lora-demo ^
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers ^
    --train_batch_size 1 ^
    --num_train_epochs 1 ^
    --learning_rate 5e-5 ^
    --gradient_checkpointing ^
    --mixed_precision fp16 ^
    --gradient_accumulation_steps 4 ^
    --resolution 512 ^
    --synthetic_num_samples 50000 ^
    --max_train_steps 20000 ^
    --checkpointing_steps 5000 ^
    --checkpoints_total_limit 5 ^
    --enable_ocr_loss ^
    --ocr_loss_weight 0.3 ^
    --enable_rtl_loss ^
    --rtl_loss_weight 0.1 ^
    --use_english_prompts ^
    --composition_mode mixed ^
    --max_phrase_length 3 ^
    --character_level_prob 0.15 ^
    --rank 8 ^
    --train_text_encoder

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Model saved in: arabic-lora-demo
echo.
echo Test with:
echo python run.py --model_type sd3 --arabic_diffusion_mode hybrid --arabic_lora_path training/arabic_diffusion/arabic-lora-demo --prompt_file prompts_arabic.txt --scheduler overshoot --use_att --num_inference_steps 28 --c 2.0
echo.
pause

