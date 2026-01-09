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
    lora_loaded = False
    if args.arabic_lora_path:
        try:
            lora_path = args.arabic_lora_path.strip()
            
            # Normalize path separators (handle Windows backslashes)
            # Replace backslashes with forward slashes for cross-platform compatibility
            lora_path_normalized = lora_path.replace('\\', '/')
            
            # Check if path has separators (definitely a local path, not HuggingFace repo)
            has_separators = ('/' in lora_path or '\\' in lora_path)
            
            # Try to find the LoRA path
            found_lora_path = None
            
            # First, try the path as-is
            if os.path.exists(lora_path):
                found_lora_path = os.path.abspath(lora_path)
            # Try normalized path
            elif os.path.exists(lora_path_normalized):
                found_lora_path = os.path.abspath(lora_path_normalized)
            # Try as relative path from current directory
            elif not os.path.isabs(lora_path) or has_separators:
                # Remove leading slash/backslash if present (Windows quirk)
                clean_path = lora_path.lstrip('/\\')
                clean_normalized = lora_path_normalized.lstrip('/\\')
                
                # Try multiple possible locations
                possible_paths = [
                    clean_path,  # Cleaned version
                    clean_normalized,  # Cleaned normalized version
                    os.path.normpath(clean_path),  # Normalized
                    lora_path,  # Original
                    lora_path_normalized,  # Original normalized
                    os.path.join("training", "arabic_diffusion", clean_path),
                    os.path.join("training", "arabic_diffusion", clean_normalized),
                    os.path.join("arabic-lora-output", clean_path),
                    os.path.join("arabic-lora-output", clean_normalized),
                ]
                
                for path in possible_paths:
                    path = os.path.normpath(path)  # Normalize path
                    if os.path.exists(path):
                        found_lora_path = os.path.abspath(path)
                        break
                    # Also check if it's a directory with safetensors file inside
                    safetensors_in_dir = os.path.join(path, "pytorch_lora_weights.safetensors")
                    if os.path.exists(safetensors_in_dir):
                        found_lora_path = os.path.abspath(path)
                        break
            
            # If we found a local path, use it
            if found_lora_path and os.path.exists(found_lora_path):
                lora_path = found_lora_path
                
                # It's a local path - use it directly
                if os.path.isdir(lora_path):
                    # If it's a directory, check for the safetensors file
                    safetensors_path = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
                    if os.path.exists(safetensors_path):
                        # Load from directory (diffusers will find the safetensors file)
                        print(f"\n{'='*60}")
                        print(f"Loading Arabic LoRA from: {lora_path}")
                        print(f"Found safetensors file: {safetensors_path}")
                        file_size = os.path.getsize(safetensors_path) / (1024 * 1024)  # MB
                        print(f"LoRA file size: {file_size:.2f} MB")
                        print(f"{'='*60}\n")
                        
                        pipe.load_lora_weights(lora_path, adapter_name="arabic")
                        print(f"✓ Successfully loaded Arabic LoRA weights")
                        
                        # Set different scales for transformer (background) vs text encoders (text)
                        transformer_scale = args.lora_scale_transformer if args.lora_scale_transformer is not None else args.lora_scale
                        text_encoder_scale = args.lora_scale_text_encoder if args.lora_scale_text_encoder is not None else (args.lora_scale * 1.5)
                        
                        # Apply selective scaling if supported
                        try:
                            if hasattr(pipe.transformer, 'set_adapters'):
                                # Try to set different scales for transformer and text encoders
                                scales_dict = {
                                    "transformer": {"down": transformer_scale, "mid": transformer_scale, "up": transformer_scale},
                                }
                                pipe.transformer.set_adapters(["arabic"], weights=scales_dict)
                                print(f"✓ Activated LoRA adapter: arabic")
                                print(f"  Transformer scale: {transformer_scale:.2f} (affects background)")
                            elif hasattr(pipe, 'set_adapters'):
                                pipe.set_adapters(["arabic"])
                                print(f"✓ Activated LoRA adapter: arabic")
                        except Exception as e:
                            # Fallback to simple activation
                            if hasattr(pipe.transformer, 'set_adapters'):
                                pipe.transformer.set_adapters(["arabic"])
                            elif hasattr(pipe, 'set_adapters'):
                                pipe.set_adapters(["arabic"])
                            print(f"✓ Activated LoRA adapter: arabic (using uniform scale)")
                        
                        # Verify LoRA was loaded
                        if hasattr(pipe.transformer, 'get_active_adapters'):
                            active_adapters = pipe.transformer.get_active_adapters()
                            print(f"✓ Active LoRA adapters: {active_adapters}")
                        elif hasattr(pipe, 'get_active_adapters'):
                            active_adapters = pipe.get_active_adapters()
                            print(f"✓ Active LoRA adapters: {active_adapters}")
                        
                        print(f"✓ LoRA scales:")
                        print(f"  - Transformer (background): {transformer_scale:.2f}")
                        print(f"  - Text encoders (text): {text_encoder_scale:.2f}")
                        print(f"  - Cross-attention (used in generation): {args.lora_scale:.2f}")
                        print(f"{'='*60}\n")
                        lora_loaded = True
                    else:
                        raise FileNotFoundError(f"LoRA weights not found in {lora_path}. Expected: {safetensors_path}")
                else:
                    # It's a file path
                    print(f"\n{'='*60}")
                    print(f"Loading Arabic LoRA from file: {lora_path}")
                    print(f"{'='*60}\n")
                    pipe.load_lora_weights(lora_path, adapter_name="arabic")
                    print(f"✓ Successfully loaded Arabic LoRA from file")
                    # Activate the LoRA adapter
                    if hasattr(pipe.transformer, 'set_adapters'):
                        pipe.transformer.set_adapters(["arabic"])
                        print(f"✓ Activated LoRA adapter: arabic")
                    elif hasattr(pipe, 'set_adapters'):
                        pipe.set_adapters(["arabic"])
                        print(f"✓ Activated LoRA adapter: arabic")
                    print(f"✓ LoRA scale: {args.lora_scale}")
                    print(f"{'='*60}\n")
                    lora_loaded = True
            else:
                # Only try HuggingFace if it doesn't have path separators (not a local path)
                if not has_separators:
                    # Try as HuggingFace repo
                    print(f"\n{'='*60}")
                    print(f"Loading Arabic LoRA from HuggingFace: {lora_path}")
                    print(f"{'='*60}\n")
                    pipe.load_lora_weights(lora_path, adapter_name="arabic")
                    print(f"✓ Successfully loaded Arabic LoRA from HuggingFace")
                    # Activate the LoRA adapter
                    if hasattr(pipe.transformer, 'set_adapters'):
                        pipe.transformer.set_adapters(["arabic"])
                        print(f"✓ Activated LoRA adapter: arabic")
                    elif hasattr(pipe, 'set_adapters'):
                        pipe.set_adapters(["arabic"])
                        print(f"✓ Activated LoRA adapter: arabic")
                    print(f"✓ LoRA scale: {args.lora_scale}")
                    print(f"{'='*60}\n")
                    lora_loaded = True
                else:
                    raise FileNotFoundError(
                        f"LoRA path not found: {lora_path}\n"
                        f"Please provide:\n"
                        f"  - An absolute path to the LoRA directory/file\n"
                        f"  - A relative path from current directory\n"
                        f"  - A HuggingFace repo ID (e.g., 'username/repo-name')"
                    )
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR: Could not load Arabic LoRA: {e}")
            print(f"{'='*60}\n")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  Continuing WITHOUT Arabic LoRA - results may be poor!\n")
    else:
        print(f"\n⚠️  No Arabic LoRA path provided - running without LoRA\n")
    
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
        
        # Prepare joint_attention_kwargs for LoRA scale if LoRA is loaded (SD3 uses joint_attention_kwargs, not cross_attention_kwargs)
        joint_attention_kwargs = None
        if lora_loaded:
            # Use the transformer scale (lower) to minimize background impact
            transformer_scale = args.lora_scale_transformer if args.lora_scale_transformer is not None else args.lora_scale
            joint_attention_kwargs = {"scale": transformer_scale}
            print(f"  Using LoRA with transformer scale: {transformer_scale:.2f} (preserves background)")
        
        output = pipe(
            prompt=prompt_text,
            num_inference_steps=args.num_inference_steps,
            height=args.img_size,
            width=args.img_size,
            guidance_scale=guidance_scale,
            generator=generator,
            use_att=args.use_att,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        image = output.images[0]
        
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
    
    # Arabic LoRA loading
    parser.add_argument(
        "--arabic_lora_path",
        type=str,
        default=None,
        help="Path to Arabic LoRA weights directory or file"
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=0.6,
        help="Scale for LoRA weights (0.0 to 1.0, default: 0.6). Higher values improve text accuracy but may affect background."
    )
    parser.add_argument(
        "--lora_scale_transformer",
        type=float,
        default=None,
        help="Separate scale for transformer LoRA (affects background). If None, uses lora_scale."
    )
    parser.add_argument(
        "--lora_scale_text_encoder",
        type=float,
        default=None,
        help="Separate scale for text encoder LoRA (affects text only). If None, uses lora_scale * 1.5."
    )
    
    args = parser.parse_args()
    
    run(args)
    