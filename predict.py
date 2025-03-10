import os
import time
import base64
import requests
import subprocess
from io import BytesIO
from time import perf_counter
from urllib.parse import urlparse
from typing import Optional

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from diffusers import (ControlNetModel, 
                    StableDiffusionControlNetInpaintPipeline,
                    StableDiffusionInpaintPipeline)


from models.diffusion import Diffusion
import shutil

MODEL_CACHE = "/tmp/huggingface_cache"
CACHE_DIR = MODEL_CACHE
DEVICE = "cuda"
DTYPE = torch.get_default_dtype()

# Define cache directory - this will be the single source of truth for models

# Model IDs and URLs
PATH_DIFFUSION_BASE_MODEL = "botp/stable-diffusion-v1-5-inpainting"
PATH_DIFFUSION_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
PATH_DIFFUSION_CONTROLNET_SEG = "lllyasviel/control_v11p_sd15_seg"
PATH_DIFFUSION_CONTROLNET_HOUGH = "lllyasviel/control_v11p_sd15_mlsd"
# PATH_MODEL_MLSD = os.getenv("MLSD_DETECTOR_MODEL", "lllyasviel/ControlNet")
# PATH_MODEL_SEGFORMER = os.getenv("HUGGINGFACE_MODEL_SEGMENT", "nvidia/segformer-b5-finetuned-ade-640-640")

# LAMA model for GAN inpainting
# LAMA_MODEL_URL = os.getenv("LAMA_MODEL_URL", "https://huggingface.co/botp/stable-diffusion-1.5-inpainting/resolve/main/big-lama.pt")
# PATH_LAMA = os.path.join(CACHE_DIR, "big-lama.pt")


def download_models():
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    model_exists = os.path.exists(os.path.join(CACHE_DIR, f"models--{PATH_DIFFUSION_CONTROLNET_SEG.replace('/', '--')}"))
    if not model_exists:
        print("Downloading ControlNet-Seg model...")
        controlnet_seg = ControlNetModel.from_pretrained(
            PATH_DIFFUSION_CONTROLNET_SEG,
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        )
        del controlnet_seg

    model_exists = os.path.exists(os.path.join(CACHE_DIR, f"models--{PATH_DIFFUSION_CONTROLNET_HOUGH.replace('/', '--')}"))
    if not model_exists:
        print("Downloading ControlNet-MLSD (Hough) model...")
        controlnet_hough = ControlNetModel.from_pretrained(
            PATH_DIFFUSION_CONTROLNET_HOUGH,
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        )
        del controlnet_hough
        
    # model_exists = os.path.exists(os.path.join(CACHE_DIR, "models--stabilityai/sd-vae-ft-mse".replace('/', '--')))
    # if not model_exists:
    #     print("Downloading base diffusion model...")
    #     # First get VAE for better precision
    #     vae = AutoencoderKL.from_pretrained(
    #         "stabilityai/sd-vae-ft-mse",
    #         torch_dtype=torch.float16,
    #         cache_dir=CACHE_DIR
    #     )
    
    model_exists = os.path.exists(os.path.join(CACHE_DIR, f"models--{PATH_DIFFUSION_BASE_MODEL.replace('/', '--')}"))
    if not model_exists:
    # if 1:
        # Get main pipeline with VAE
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            PATH_DIFFUSION_BASE_MODEL,
             variant='fp16',
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR,
            # safety_checker=None,
            
            # use_safetensors=False
        )
        pipe.safety_checker.threshold = 0.9
        print("==========")
        del pipe
    
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



def download_image(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image"""
    # try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))
    # except Exception as e:
    #     raise ValueError(f"Failed to download image from {url}: {str(e)}")


def resize_image(
    input_image: Image.Image | np.ndarray,
    resolution: int,
    return_dim_only: bool = False,
) -> tuple[int, int] | Image.Image:
    """Resize image to target resolution while maintaining aspect ratio"""
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)

    H, W = input_image.shape[:2]
    scale = float(resolution) / min(H, W)
    new_H = int(np.round(H * scale / 64.0)) * 64
    new_W = int(np.round(W * scale / 64.0)) * 64

    if return_dim_only:
        return new_H, new_W

    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(input_image, (new_W, new_H), interpolation=interpolation)
    return Image.fromarray(resized)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Initialize the model and load weights"""
        self.max_input_resolution = 1024
        
        # Model configuration
        self.base_model_name = "stable-diffusion-v1-5/stable-diffusion-inpainting"
        self.controlnet_seg_name = "lllyasviel/control_v11p_sd15_seg"
        self.controlnet_hough_name = "lllyasviel/control_v11p_sd15_mlsd"
        # self.scheduler_name = "DPMSolverMultistepScheduler"
                    

        # self._setup_model_cache()

        self._setup_torch_environment()

        download_models()

        self._load_diffusion_model()


    def _setup_torch_environment(self):
        """Configure torch settings"""
        torch.set_default_dtype(DTYPE)
        torch.set_default_device(DEVICE)

    def _load_diffusion_model(self):
        """Load the diffusion model"""
        self.diffusion = Diffusion(
            base_model_name=self.base_model_name,
            controlnet_seg_name=self.controlnet_seg_name,
            controlnet_hough_name=self.controlnet_hough_name,
            # scheduler_name=self.scheduler_name,
            # force_cpu=DEVICE == "cpu"
        )
        # Initialize the model
        self.diffusion._initialize()

    def _log_timing(self, step: str, start_time: float):
        """Log the execution time of a step"""
        duration = perf_counter() - start_time
        print(f"{step} completed in {duration:.2f} seconds")


    def _prepare_images(
        self, image_url: str, mask_url: str, control_url: str = None, control_type: str = None
    ) -> tuple:
        """Prepare input images for processing from URLs"""
        start_time = perf_counter()

        try:
            image = download_image(image_url).convert("RGB")
            mask_image = download_image(mask_url).convert("L")
            
            control_image = None
            if control_url and control_type:
                control_image = download_image(control_url)
                if control_type == "segmentation_mask":
                    control_image = control_image.convert("RGB")
                elif control_type == "straight_line":
                    control_image = control_image.convert("L")
        except ValueError as e:
            raise ValueError(f"Failed to load images: {str(e)}")

        # Resize images if they exceed max resolution
        do_resize = min(image.size) > self.max_input_resolution
        if do_resize:
            image = resize_image(image, self.max_input_resolution)
            
        # Ensure mask has same dimensions as image
        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size)
            
        # Ensure control image has same dimensions as image
        if control_image and control_image.size != image.size:
            control_image = control_image.resize(image.size)

        self._log_timing("Image preparation", start_time)
        return image, mask_image, control_image, do_resize

    def predict(
        self,
        image_url: str = Input(
            description="URL of the input image to inpaint"
        ),
        mask_url: str = Input(
            description="URL of the mask image for inpainting"
        ),
        prompt: str = Input(
            description="Text prompt for guidance",
            
        ),
        negative_prompt: str = Input(
            description="Negative text prompt for guidance",
            default=None
            
        ),
        control_url: str = Input(
            description="URL of the control image (optional)",
            default=None
        ),
        control_type: str = Input(
            description="Type of control to use",
            choices=["segmentation_mask", "straight_line", "none"],
            default="none",
        ),
        num_images: int = Input(
            description="Number of images to generate",
            ge=1,
            le=4,
            default=1,
        ),
        num_steps: int = Input(
            description="Number of inference steps",
            ge=10,
            le=100,
            default=30,
            
        ),
        guidance_scale: float = Input(
            description="Guidance scale for text prompt",
            ge=1.0,
            le=20.0,
            default=7.5
            
        ),
        control_guidance_start: float = Input(
            description="Control guidance start value",
            ge=0.0,
            le=1.0,
            default=0.0
        ),
        control_guidance_end: float = Input(
            description="Control guidance end value",
            ge=0.0,
            le=1.0,
            default=1.0,
           
        ),
        control_conditioning_scale: float = Input(
            description="Control conditioning scale",
            ge=0.0,
            le=2.0,
            default=1.0,
           
        ),
        strength: float = Input(
            description="How strong the inpainting should be (0-1)",
            ge=0.0,
            le=1.0,
            default=0.8,
 
        ),
        seed: int = Input(
            description="Random seed (-1 for random)",
            default=22,
        ),
        output_format: str = Input(
            description="Output image format",
            choices=["jpg", "png"],
            default="png",
        ),
        output_quality: int = Input(
            description="Output image quality (0-100)",
            ge=0,
            le=100,
            default=95,
        ),
    ) -> Path:
        """Process image with diffusion-based inpainting and control guidance"""
        total_start = perf_counter()

        # Validate URLs
        for url in [image_url, mask_url]:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError(f"Invalid URL: {url}")
                
        if control_url:
            parsed = urlparse(control_url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError(f"Invalid control URL: {control_url}")

        # Prepare images
        image, mask_image, control_image, _ = self._prepare_images(
            image_url, mask_url, control_url, control_type
        )

        # Padding for mask crop (can be adjusted based on needs)
        padding_mask_crop = None  # or a specific value like 32

        # Process the image with diffusion model
        inpaint_start = perf_counter()
        # print(type(strength))
        output_images = self.diffusion._execute(
            input_image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            control_conditioning_scale=control_conditioning_scale,
            strength=strength,
            mask_image=mask_image,
            control_type=control_type,
            control_image=control_image,
            seed=seed,
            padding_mask_crop=padding_mask_crop
        )
        self._log_timing("Diffusion inpainting", inpaint_start)

        # Save the output image
        encode_start = perf_counter()
        output_path = "output_image.jpg" if output_format == "jpg" else "output_image.png"
        format = "JPEG" if output_format == "jpg" else "PNG"
        
        if num_images > 1:
            # If multiple images, save the first one
            output_images[0].save(
                output_path,
                format=format,
                quality=output_quality if output_format == "jpg" else None
            )
            
            # Save additional images with suffixes
            for i in range(1, len(output_images)):
                additional_path = f"output_image_{i}.{output_format}"
                output_images[i].save(
                    additional_path,
                    format=format,
                    quality=output_quality if output_format == "jpg" else None
                )
        else:
            # Just save the single image
            output_images[0].save(
                output_path,
                format=format,
                quality=output_quality if output_format == "jpg" else None
            )

        self._log_timing("Image encoding", encode_start)
        self._log_timing("Total processing", total_start)
        
        return Path(output_path)
        
    def __del__(self):
        """Clean up resources when the predictor is destroyed"""
        # if hasattr(self, 'diffusion'):
        print("==")
            # self.diffusion._finalize()

# if __name__=="__main__":
#     image_url = f"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
#     mask_url = f"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
#     control_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"

#     # Run predictor
#     predictor = Predictor()
#     predictor.setup()
    
#     result = predictor.predict(
#         image_url=image_url,
#         mask_url=mask_url,
#         prompt="empty",
#         negative_prompt="blur",
#         control_url=control_url,
#         control_type="segmentation_mask",
#         num_images=1,
#         num_steps=30,
#         guidance_scale=7.5,
#         control_guidance_start=0.0,
#         control_guidance_end=1.0,
#         control_conditioning_scale=1.0,
#         strength=0.8,
#         seed=22,
#         output_format="png",
#         output_quality=95

#     )
    
#     print(f"Processing complete! Result saved to: {result}")