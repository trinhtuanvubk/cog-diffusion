import os
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download
from torch.hub import download_url_to_file
from diffusers import AutoencoderKL, DiffusionPipeline, ControlNetModel
import traceback


# Define cache directory - this will be the single source of truth for models
CACHE_DIR = "/tmp/huggingface_cache"

# Model IDs and URLs
PATH_DIFFUSION_BASE_MODEL = os.getenv("HUGGINGFACE_MODEL_PIPELINE", "botp/stable-diffusion-v1-5-inpainting")
PATH_DIFFUSION_CONTROLNET_SEG = os.getenv("HUGGINGFACE_MODEL_CONTROLNET_SEG", "lllyasviel/control_v11p_sd15_seg")
PATH_DIFFUSION_CONTROLNET_HOUGH = os.getenv("HUGGINGFACE_MODEL_CONTROLNET_MLSD", "lllyasviel/control_v11p_sd15_mlsd")
# PATH_MODEL_MLSD = os.getenv("MLSD_DETECTOR_MODEL", "lllyasviel/ControlNet")
# PATH_MODEL_SEGFORMER = os.getenv("HUGGINGFACE_MODEL_SEGMENT", "nvidia/segformer-b5-finetuned-ade-640-640")

# LAMA model for GAN inpainting
# LAMA_MODEL_URL = os.getenv("LAMA_MODEL_URL", "https://huggingface.co/botp/stable-diffusion-1.5-inpainting/resolve/main/big-lama.pt")
# PATH_LAMA = os.path.join(CACHE_DIR, "big-lama.pt")


def download_models():
    try:
        # Reset cache if needed
        if os.path.exists(CACHE_DIR) and os.getenv("RESET_CACHE", "0") == "1":
            print(f"Removing existing cache directory: {CACHE_DIR}")
            shutil.rmtree(CACHE_DIR)
        
        # Create cache directory
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        
        print("Downloading ControlNet-Seg model...")
        controlnet_seg = ControlNetModel.from_pretrained(
            PATH_DIFFUSION_CONTROLNET_SEG,
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        )
        
        print("Downloading ControlNet-MLSD (Hough) model...")
        controlnet_hough = ControlNetModel.from_pretrained(
            PATH_DIFFUSION_CONTROLNET_HOUGH,
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        )
        
        
        print("Downloading base diffusion model...")
        # First get VAE for better precision
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        )
        
        # Get main pipeline with VAE
        pipe = DiffusionPipeline.from_pretrained(
            PATH_DIFFUSION_BASE_MODEL,
            vae=vae,
            torch_dtype=torch.float16,
            variant='fp16',
            cache_dir=CACHE_DIR,
            safety_checker=None
        )
        
        # Verify models are in cache
        model_files = os.listdir(CACHE_DIR)
        
        return {
            "status": "success",
            "message": "All models downloaded successfully to cache",
            "cache_dir": CACHE_DIR,
            "models": {
                "base_model": PATH_DIFFUSION_BASE_MODEL,
                "controlnet_seg": PATH_DIFFUSION_CONTROLNET_SEG,
                "controlnet_hough": PATH_DIFFUSION_CONTROLNET_HOUGH,
            }
        }
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        return {
            "status": "error",
            "message": str(e),
            "traceback": error_traceback
        }

