# Step-by-Step Training Guide

## Quick Checklist

- [ ] Step 1: Verify vocabulary file exists
- [ ] Step 2: Activate conda environment
- [ ] Step 3: Navigate to training directory
- [ ] Step 4: Run training command
- [ ] Step 5: Monitor training progress
- [ ] Step 6: Test checkpoints (optional during training)
- [ ] Step 7: Use final model after training

---

## Step 1: Verify Vocabulary File ✅

**Check if you have the large vocabulary file:**

```bash
# From Fine-tune directory
python -c "with open('training/arabic_diffusion/Output/arabic_words.txt', 'r', encoding='utf-8') as f: words = [l.strip() for l in f if l.strip()]; print(f'Found {len(words)} Arabic words')"
```

**Expected output:** `Found 11517 Arabic words` (or similar large number)

**If file doesn't exist or has few words:**
- Check: `training/arabic_diffusion/Output/arabic_words.txt`
- Should have 11,517+ words
- If missing, you need to create/obtain this file first

---

## Step 2: Activate Conda Environment 🐍

```bash
conda activate amo2
```

**Verify you're in the right environment:**
```bash
python --version
# Should show Python 3.x
```

---

## Step 3: Navigate to Training Directory 📁

```bash
cd training\arabic_diffusion
```

**Verify you're in the right place:**
```bash
dir
# Should see: train_lora.py, dataset.py, etc.
```

---

## Step 4: Run Training Command 🚀

**Copy and paste this entire command:**

```bash
python train_lora.py ^
    --arabic_words_file Output/arabic_words.txt ^
    --output_dir arabic-lora-production ^
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers ^
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
```

**What this does:**
- Uses 11,517 words (large vocabulary)
- Trains for 100,000 steps (better quality)
- Saves checkpoints every 5,000 steps
- Uses English prompts (production mode)
- Character + word + phrase training (flexible)
- OCR loss enabled (better accuracy)

**Press Enter** and training will start!

---

## Step 5: Monitor Training Progress 📊

**You'll see output like:**

```
Steps:   5%|██▌                    | 5000/100000 [02:15<45:30, 34.7it/s, diffusion_loss=0.001, ocr_loss=0.05, total_loss=0.006]
```

**What to watch:**
- **Steps**: Progress (5000/100000 = 5% done)
- **diffusion_loss**: Should decrease over time
- **ocr_loss**: Should decrease (text getting better)
- **total_loss**: Overall training loss

**Checkpoints are saved:**
- Every 5,000 steps
- Location: `arabic-lora-production/checkpoint-5000/`, `checkpoint-10000/`, etc.
- Final model: `arabic-lora-production/pytorch_lora_weights.safetensors`

---

## Step 6: Test Checkpoints (Optional) 🧪

**You can test during training!**

**After 50,000 steps (about 2-3 days):**

```bash
# From Fine-tune directory
python run.py ^
    --model_type sd3 ^
    --arabic_diffusion_mode hybrid ^
    --arabic_lora_path training/arabic_diffusion/arabic-lora-production/checkpoint-50000 ^
    --prompt_file prompts_arabic.txt ^
    --scheduler overshoot ^
    --use_att ^
    --num_inference_steps 28 ^
    --c 2.0
```

**Check results:**
- Are images generating?
- Is Arabic text readable?
- If good → can use this checkpoint
- If not → wait for more training

---

## Step 7: Use Final Model After Training ✅

**After 100,000 steps complete (about 5-7 days):**

```bash
# From Fine-tune directory
python run.py ^
    --model_type sd3 ^
    --arabic_diffusion_mode hybrid ^
    --arabic_lora_path training/arabic_diffusion/arabic-lora-production ^
    --prompt_file prompts_arabic.txt ^
    --scheduler overshoot ^
    --use_att ^
    --num_inference_steps 28 ^
    --c 2.0
```

**The model is ready!** 🎉

---

## Training Timeline ⏱️

| Step | Time | What Happens |
|------|------|--------------|
| 0-10k | ~12 hours | Early learning |
| 10k-50k | ~2-3 days | Significant improvement |
| 50k-75k | ~1-2 days | Refinement |
| 75k-100k | ~1-2 days | Final polish |
| **Total** | **5-7 days** | Complete training |

**You can test at:**
- 50k steps (if you want to test early)
- 75k steps (good quality)
- 100k steps (best quality)

---

## Troubleshooting 🔧

### Problem: "File not found: Output/arabic_words.txt"
**Solution:** Check the path. From `training/arabic_diffusion/`, the file should be at `Output/arabic_words.txt`

### Problem: "Out of memory" error
**Solution:** Reduce batch size or resolution:
- `--train_batch_size 1` (already set)
- `--resolution 384` (instead of 512)

### Problem: Training is too slow
**Solution:** This is normal. 100k steps takes 5-7 days. You can:
- Test at 50k steps (2-3 days)
- Reduce to 50k steps if needed: `--max_train_steps 50000`

### Problem: Want to resume training
**Solution:** Training automatically saves checkpoints. To resume:
```bash
python train_lora.py ... --resume_from_checkpoint arabic-lora-production/checkpoint-50000
```

---

## Quick Start (Copy-Paste Ready) 📋

```bash
# Step 1: Activate environment
conda activate amo2

# Step 2: Go to training directory
cd training\arabic_diffusion

# Step 3: Verify vocabulary (optional check)
python -c "with open('Output/arabic_words.txt', 'r', encoding='utf-8') as f: words = [l.strip() for l in f if l.strip()]; print(f'Found {len(words)} words')"

# Step 4: Start training
python train_lora.py --arabic_words_file Output/arabic_words.txt --output_dir arabic-lora-production --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers --train_batch_size 1 --num_train_epochs 1 --learning_rate 5e-5 --gradient_checkpointing --mixed_precision fp16 --gradient_accumulation_steps 4 --resolution 512 --synthetic_num_samples 500000 --max_train_steps 100000 --checkpointing_steps 5000 --checkpoints_total_limit 10 --enable_ocr_loss --ocr_loss_weight 0.2 --enable_rtl_loss --rtl_loss_weight 0.1 --use_english_prompts --composition_mode mixed --max_phrase_length 5 --character_level_prob 0.1 --rank 8 --train_text_encoder
```

**That's it! Training will start and run for 5-7 days.**

---

## Summary

1. ✅ Verify vocabulary file (11,517 words)
2. ✅ Activate `amo2` environment
3. ✅ Go to `training\arabic_diffusion`
4. ✅ Run training command
5. ⏳ Wait 5-7 days (or test at 50k steps)
6. ✅ Use final model!

**Ready to start?** Follow steps 1-4 above! 🚀

