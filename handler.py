import os
import torch
import runpod
import base64
import io
from diffusers import FluxPipeline, StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from PIL import Image

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

MODELS_DIR = "/models"
CHECKPOINT_DIR = f"{MODELS_DIR}/checkpoints"
LORA_DIR = f"{MODELS_DIR}/loras"
VAE_DIR = f"{MODELS_DIR}/vae"

# Global pipeline variable
pipe = None
pipeline_info = {}

def load_models():
    global pipe, pipeline_info
    
    print("Loading models...")
    
    # Check if models exist in the baked-in directory
    checkpoint_path = None
    if os.path.exists(CHECKPOINT_DIR):
        files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.safetensors')]
        if files:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, files[0])
            print(f"Found baked-in checkpoint: {checkpoint_path}")
            
    vae_path = None
    if os.path.exists(VAE_DIR):
        files = [f for f in os.listdir(VAE_DIR) if f.endswith('.safetensors')]
        if files:
            vae_path = os.path.join(VAE_DIR, files[0])
            print(f"Found baked-in VAE: {vae_path}")
            
    lora_paths = []
    if os.path.exists(LORA_DIR):
        lora_paths = [os.path.join(LORA_DIR, f) for f in os.listdir(LORA_DIR) if f.endswith('.safetensors')]
        print(f"Found {len(lora_paths)} baked-in LoRAs")

    if not checkpoint_path:
        print("No baked-in checkpoint found! Check build process.")
        return False

    # Load VAE
    vae = None
    if vae_path:
        try:
            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
        except Exception as e:
            print(f"Error loading VAE: {e}")

    # Determine model type (simple heuristic or env var)
    # Default to SDXL as per original script default
    model_type = os.environ.get("MODEL_TYPE", "SDXL") 
    
    print(f"Loading {model_type} pipeline from {checkpoint_path}...")
    
    if model_type == "Flux":
        pipe = FluxPipeline.from_single_file(
            checkpoint_path,
            vae=vae,
            torch_dtype=dtype
        )
    else: # SDXL
        pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint_path,
            vae=vae,
            torch_dtype=dtype
        )
        
    pipe = pipe.to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        
    # Load LoRAs
    if lora_paths:
        print("Loading LoRAs...")
        loaded_loras = []
        for i, lora_path in enumerate(lora_paths):
            try:
                adapter_name = f"lora_{i+1}"
                pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                loaded_loras.append(adapter_name)
                print(f"Loaded LoRA: {lora_path}")
            except Exception as e:
                print(f"Error loading LoRA {i+1}: {e}")
                
        # Set adapters active
        if loaded_loras:
            # Default scale 0.8 as per script
            scales = [0.8] * len(loaded_loras)
            pipe.set_adapters(loaded_loras, adapter_weights=scales)

    pipeline_info = {
        "model_type": model_type,
        "loaded": True
    }
    print("Models loaded successfully!")
    return True

# Initialize models at startup
load_models()

def handler(job):
    job_input = job["input"]
    
    if not pipeline_info.get("loaded"):
        return {"error": "Pipeline not loaded"}

    # Extract parameters with defaults from imagen.py
    prompt = job_input.get("prompt", "a beautiful landscape with mountains and a lake, highly detailed, 8k, photorealistic")
    negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distorted, ugly, bad anatomy")
    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    steps = job_input.get("steps", 30)
    cfg_scale = job_input.get("cfg_scale", 5.5)
    seed = job_input.get("seed", None)
    scheduler_type = job_input.get("scheduler", "Euler a")
    clip_skip = job_input.get("clip_skip", 2) # SDXL only

    # Configure Scheduler
    if scheduler_type == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++"
        )
        
    generator = None
    if seed:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        # Generate a seed if none provided so we can return it
        seed = torch.seed()
        generator = torch.Generator(device=device).manual_seed(seed)

    print(f"Generating: Prompt='{prompt[:50]}...', Steps={steps}, Seed={seed}")

    gen_params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "generator": generator
    }

    if pipeline_info["model_type"] == "SDXL" and clip_skip > 1:
        gen_params["clip_skip"] = clip_skip

    try:
        output = pipe(**gen_params)
        image = output.images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "image": img_str,
            "seed": seed,
            "params": {
                "width": width, 
                "height": height,
                "steps": steps,
                "cfg": cfg_scale,
                "model": pipeline_info["model_type"]
            }
        }
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
