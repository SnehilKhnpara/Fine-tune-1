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
                        print(f"\n{'='*60}")
                        print(f"Loading Arabic LoRA from: {lora_path}")
                        print(f"Found safetensors file: {safetensors_path}")
                        file_size = os.path.getsize(safetensors_path) / (1024 * 1024)  # MB
                        print(f"LoRA file size: {file_size:.2f} MB")
                        print(f"{'='*60}\n")
                        
                        pipe.load_lora_weights(lora_path, adapter_name="arabic")
                        print(f"✓ Successfully loaded Arabic LoRA weights")
                        
                        # Activate the LoRA adapter
                        if hasattr(pipe.transformer, 'set_adapters'):
                            pipe.transformer.set_adapters(["arabic"])
                            print(f"✓ Activated LoRA adapter: arabic")
                        elif hasattr(pipe, 'set_adapters'):
                            pipe.set_adapters(["arabic"])
                            print(f"✓ Activated LoRA adapter: arabic")
                        
                        # Verify LoRA was loaded by checking if adapter is in the list
                        if hasattr(pipe.transformer, 'get_active_adapters'):
                            active_adapters = pipe.transformer.get_active_adapters()
                            print(f"✓ Active LoRA adapters: {active_adapters}")
                        elif hasattr(pipe, 'get_active_adapters'):
                            active_adapters = pipe.get_active_adapters()
                            print(f"✓ Active LoRA adapters: {active_adapters}")
                        
                        print(f"✓ LoRA scale: {args.lora_scale}")
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
        
        # Prepare cross_attention_kwargs for LoRA scale if LoRA is loaded
        cross_attention_kwargs = None
        if lora_loaded:
            cross_attention_kwargs = {"scale": args.lora_scale}
            print(f"  Using LoRA with scale: {args.lora_scale}")
        
        output = pipe(
            prompt=prompt_text,
            num_inference_steps=args.num_inference_steps,
            height=args.img_size,
            width=args.img_size,
            guidance_scale=guidance_scale,
            generator=generator,
            use_att=args.use_att,
            cross_attention_kwargs=cross_attention_kwargs,
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
        default=1.0,
        help="Scale for LoRA weights (0.0 to 1.0, default: 1.0)"
    )
    
    args = parser.parse_args()
    
    run(args)
    