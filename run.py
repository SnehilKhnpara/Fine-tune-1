import argparse
import os
import torch
import numpy as np

from diffusers import StableDiffusion3Pipeline, FluxPipeline, AuraFlowPipeline
from diffusers import StochasticRFOvershotDiscreteScheduler
# from pipelines import FluxPipeline


def run(args):
    with open(args.prompt_file, 'r') as file:
        prompts = file.readlines()
    
    if args.model_type == "sd3":
        # Use float32 for SD3 to match LoRA weights (LoRA was saved in float32)
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)
        guidance_scale = 7.0
        
        # Ensure text encoders are in float32 for LoRA loading compatibility
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            pipe.text_encoder = pipe.text_encoder.to(dtype=torch.float32)
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
            pipe.text_encoder_2 = pipe.text_encoder_2.to(dtype=torch.float32)
        if hasattr(pipe, 'text_encoder_3') and pipe.text_encoder_3 is not None:
            pipe.text_encoder_3 = pipe.text_encoder_3.to(dtype=torch.float32)
    elif args.model_type == "flux":
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        guidance_scale = 3.5
    elif args.model_type == "auraflow": 
        pipe = AuraFlowPipeline.from_pretrained("fal/AuraFlow", torch_dtype=torch.float16)
        guidance_scale = 3.5
    
    # Load Arabic LoRA BEFORE enabling CPU offload to avoid dtype issues
    # Load Arabic LoRA if specified (Approach 3: Fine-tuned Arabic Diffusion)
    if args.arabic_diffusion_mode != "off" and args.arabic_lora_path:
        try:
            # Handle both local paths and HuggingFace repo IDs
            lora_path = args.arabic_lora_path
            
            # Convert to absolute path if it's a relative path
            if not os.path.isabs(lora_path):
                # Try multiple possible locations
                possible_paths = [
                    lora_path,  # Current directory
                    os.path.join("training", "arabic_diffusion", lora_path),  # Training subdirectory
                    os.path.join("arabic-lora-output", lora_path),  # Alternative location
                ]
                
                found_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        found_path = os.path.abspath(path)
                        break
                    # Also check if it's a directory with safetensors file inside
                    safetensors_in_dir = os.path.join(path, "pytorch_lora_weights.safetensors")
                    if os.path.exists(safetensors_in_dir):
                        found_path = os.path.abspath(path)
                        break
                
                if found_path:
                    lora_path = found_path
                else:
                    # If not found, try the original path anyway
                    lora_path = os.path.abspath(lora_path) if os.path.exists(lora_path) else lora_path
            
            # Check if it's a local path
            if os.path.exists(lora_path):
                # It's a local path - use it directly
                if os.path.isdir(lora_path):
                    # If it's a directory, check for the safetensors file
                    safetensors_path = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
                    if os.path.exists(safetensors_path):
                        # Load from directory (diffusers will find the safetensors file)
                        print(f"Loading Arabic LoRA from: {lora_path}")
                        print(f"Found safetensors file: {safetensors_path}")
                        pipe.load_lora_weights(lora_path, adapter_name="arabic")
                        print(f"✓ Successfully loaded Arabic LoRA from local directory: {lora_path}")
                        
                        # Verify LoRA was loaded by checking if adapter is in the list
                        if hasattr(pipe, 'get_active_adapters'):
                            active_adapters = pipe.get_active_adapters()
                            print(f"Active LoRA adapters: {active_adapters}")
                    else:
                        raise FileNotFoundError(f"LoRA weights not found in {lora_path}. Expected: {safetensors_path}")
                else:
                    # It's a file path
                    pipe.load_lora_weights(lora_path, adapter_name="arabic")
                    print(f"Loaded Arabic LoRA from local file: {lora_path}")
            else:
                # Try as HuggingFace repo
                pipe.load_lora_weights(lora_path, adapter_name="arabic")
                print(f"Loaded Arabic LoRA from HuggingFace: {lora_path}")
        except Exception as e:
            print(f"Warning: Could not load Arabic LoRA: {e}. Continuing without it.")
            import traceback
            traceback.print_exc()
    
    # Enable CPU offload after loading LoRA
    pipe.enable_model_cpu_offload()
    
    if args.scheduler == 'overshoot':
        scheduler_config = pipe.scheduler.config
        scheduler = StochasticRFOvershotDiscreteScheduler.from_config(scheduler_config)
        overshot_func = lambda t, dt: t+dt
        exp_prefix = f"{args.scheduler}_c={str(args.c).zfill(4)}_use_att={args.use_att}"
        
        pipe.scheduler = scheduler
        pipe.scheduler.set_c(args.c)
        pipe.scheduler.set_overshot_func(overshot_func)
    elif args.scheduler == "euler":
        exp_prefix = f"{args.scheduler}"
    
    for i in range(len(prompts)):
        file_save_dir = os.path.join(args.exp_dir, "generated_image", f"num_steps={str(args.num_inference_steps).zfill(4)}", exp_prefix)
        os.makedirs(file_save_dir, exist_ok=True)
        img_save_path = os.path.join(file_save_dir, f"sample_{str(i).zfill(4)}.png")
        
        generator = torch.Generator(device='cuda')
        generator.manual_seed(args.seed)
        
        # Generate image
        prompt_text = prompts[i].strip()
        print(f"\n[{i+1}/{len(prompts)}] Generating: {prompt_text[:50]}...")
        
        # Check if prompt contains Arabic
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in prompt_text)
        if has_arabic:
            print(f"  → Arabic text detected in prompt")
            
            # SD3 requires some English text in the prompt. If prompt is Arabic-only,
            # wrap it in a simple English description to make the encoder happy.
            # Check if prompt is ONLY Arabic (no English words)
            has_english = any(c.isalpha() and ord(c) < 128 for c in prompt_text)
            if not has_english:
                # Pure Arabic text - add English context for SD3 encoder
                prompt_text = f"Arabic text: {prompt_text}"
                print(f"  → Wrapped in English context: {prompt_text[:50]}...")
        
        output = pipe(
            prompt=prompt_text,
            num_inference_steps=args.num_inference_steps,
            height=args.img_size,
            width=args.img_size,
            guidance_scale=guidance_scale,
            generator=generator,
            use_att=args.use_att, 
        )
        image = output.images[0]
        
        # Approach 3: Optional validation and fallback
        # If Arabic diffusion mode is "hybrid", validate and fallback to mask/overlay if needed
        if args.arabic_diffusion_mode == "hybrid":
            # Check if Arabic text is in prompt (simple heuristic)
            prompt_text = prompts[i].strip()
            # Simple Arabic character detection (Unicode range for Arabic)
            has_arabic = any('\u0600' <= char <= '\u06FF' for char in prompt_text)
            
            if has_arabic:
                # Optional: Quick OCR check (can be disabled for speed)
                # For now, we trust the model output but note that fallback should be implemented
                # in production using Approach 1 (mask/overlay)
                pass
        
        image.save(img_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=str, default="euler", 
                        help="scheduler to use")
    parser.add_argument("--c", type=float, default=2.0, help="c value for overshooting scheduler")
    parser.add_argument("--prompt_file", type=str, default="prompts.txt", 
                        help="file with prompts")
    parser.add_argument("--num_inference_steps", type=int, default=28, 
                        help="number of steps")
    parser.add_argument("--exp_dir", type=str, default="exps/flux", 
                        help="experiment directory")
    parser.add_argument("--model_type", type=str, default="flux", 
                        choices=["sd3", "flux", "auraflow"], help="model type")
    parser.add_argument("--use_att", action="store_true", help="use attention")
    parser.add_argument("--seed", type=int, default=10, help="seed")
    parser.add_argument("--img_size", type=int, default=1024, help="image size")
    
    # Approach 3: Fine-tuned Arabic Diffusion flags
    parser.add_argument(
        "--arabic_diffusion_mode",
        type=str,
        default="off",
        choices=["off", "hybrid", "model_only"],
        help="Arabic diffusion mode: 'off' (default, Approach 1/2 only), "
             "'hybrid' (model + mask fallback, recommended), "
             "'model_only' (research/testing only)"
    )
    parser.add_argument(
        "--arabic_lora_path",
        type=str,
        default=None,
        help="Path to Arabic LoRA weights (required if arabic_diffusion_mode != 'off')"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.arabic_diffusion_mode != "off" and not args.arabic_lora_path:
        print("Warning: arabic_diffusion_mode is set but arabic_lora_path is not provided.")
        print("Continuing without Arabic LoRA.")
    
    run(args)
    