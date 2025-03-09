import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
from .diffusion_cropmask_wrapper import InpaintingCropWrapper

CACHE_DIR = "/tmp/huggingface_cache"

class Diffusion():

    def __init__(self, base_model_name=None, controlnet_seg_name=None, controlnet_hough_name=None, scheduler_name=None,
                 force_cpu=False):
        super().__init__(force_cpu)
        self.base_model_name = base_model_name
        self.weight_type = torch.float32 if self.device == 'cpu' else torch.float16
        self.controlnet_seg_name = controlnet_seg_name
        self.controlnet_hough_name = controlnet_hough_name
        self.scheduler_name = scheduler_name

    def _initialize(self):
        # controlnet
        self.logger.info("Loading controlnet seg from: " + self.controlnet_seg_name)
        controlnet_seg = ControlNetModel.from_pretrained(self.controlnet_seg_name, torch_dtype=self.weight_type, cache_dir=CACHE_DIR, local_files_only=True)
        self.logger.info("Loading controlnet hough from: " + self.controlnet_hough_name)
        controlnet_hough = ControlNetModel.from_pretrained(self.controlnet_seg_name, torch_dtype=self.weight_type, cache_dir=CACHE_DIR, local_files_only=True)
        # pipelines
        self.pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(self.base_model_name,
                                                                           torch_dtype=self.weight_type, cache_dir=CACHE_DIR, local_files_only=True)
        self.pipe_inpaint.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe_inpaint.scheduler.config)
        self.pipe_inpaint_hough = StableDiffusionControlNetInpaintPipeline.from_pipe(self.pipe_inpaint,
                                                                                     controlnet=controlnet_hough)
        self.pipe_inpaint_seg = StableDiffusionControlNetInpaintPipeline.from_pipe(self.pipe_inpaint,
                                                                                   controlnet=controlnet_seg)

        self.pipe_inpaint.to(self.device)
        self.pipe_inpaint_hough.to(self.device)
        self.pipe_inpaint_seg.to(self.device)

        # Wrap the pipeline with cropping functionality
        self.pipe_inpaint = InpaintingCropWrapper(self.pipe_inpaint)
        self.pipe_inpaint_hough = InpaintingCropWrapper(self.pipe_inpaint_hough)
        self.pipe_inpaint_seg = InpaintingCropWrapper(self.pipe_inpaint_seg)

    def _execute(self, input_image: Image.Image,
                 prompt: str,
                 negative_prompt: str,
                 num_images: int,
                 num_steps: int,
                 guidance_scale: float,
                 control_guidance_start: float,
                 control_guidance_end: float,
                 control_conditioning_scale: float,
                 strength: float,
                 mask_image: Image.Image,
                 control_type=None,
                 control_image: Image.Image = None,
                 seed: int = -1,
                 padding_mask_crop=None):
        if seed == -1:
            seed = np.random.randint(0, np.iinfo(np.int64).max)
        generator = torch.Generator().manual_seed(seed)
        w, h = input_image.size

        if control_image and control_type == 'straight_line':
            assert control_image is not None
            # noinspection PyCallingNonCallable
            return self.pipe_inpaint_hough(prompt=prompt,
                                           negative_prompt=negative_prompt,
                                           guidance_scale=guidance_scale,
                                           control_guidance_start=control_guidance_start,
                                           control_guidance_end=control_guidance_end,
                                           controlnet_conditioning_scale=control_conditioning_scale,
                                           num_images_per_prompt=num_images,
                                           num_inference_steps=num_steps,
                                           generator=generator,
                                           control_image=control_image,
                                           image=input_image,
                                           mask_image=mask_image,
                                           strength=strength,
                                           padding_mask_crop=padding_mask_crop,
                                           width=w,
                                           height=h).images
        elif control_image and control_type == 'segmentation_mask':
            assert control_image is not None
            # noinspection PyCallingNonCallable
            return self.pipe_inpaint_seg(prompt=prompt,
                                         negative_prompt=negative_prompt,
                                         guidance_scale=guidance_scale,
                                         control_guidance_start=control_guidance_start,
                                         control_guidance_end=control_guidance_end,
                                         controlnet_conditioning_scale=control_conditioning_scale,
                                         num_images_per_prompt=num_images,
                                         num_inference_steps=num_steps,
                                         generator=generator,
                                         control_image=control_image,
                                         image=input_image,
                                         mask_image=mask_image,
                                         strength=strength,
                                         padding_mask_crop=padding_mask_crop,
                                         width=w,
                                         height=h).images
        elif control_type is None:
            return self.pipe_inpaint(prompt=prompt,
                                     negative_prompt=negative_prompt,
                                     guidance_scale=guidance_scale,
                                     num_images_per_prompt=num_images,
                                     num_inference_steps=num_steps,
                                     generator=generator,
                                     control_image=control_image,
                                     image=input_image,
                                     mask_image=mask_image,
                                     strength=strength,
                                     padding_mask_crop=padding_mask_crop,
                                     width=w,
                                     height=h).images
        else:
            raise Exception('Unsupported control type')

    def _finalize(self):
        del self.pipe_inpaint_hough, self.pipe_inpaint_seg, self.pipe_inpaint
