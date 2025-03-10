from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
)
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from dataclasses import dataclass

from loguru import logger
# Constants
INPAINTING_MODEL = "botp/stable-diffusion-v1-5-inpainting"


class InpaintingCropWrapper:
    """Wrapper that adds automatic cropping functionality to any inpainting pipeline"""

    # Constants for image processing
    MIN_REGION_SIZE = 512  # Minimum size for processing regions
    MASK_DILATION_SIZE = 20  # Number of pixels to dilate mask
    BLEND_FEATHER_SIZE = 30  # Size of feathering for blending edges
    EDGE_PADDING_SIZE = 20  # Minimum padding from image edges

    def __init__(self, pipe):
        self.pipe = pipe
        self._original_call = pipe.__call__

    @staticmethod
    def get_bounding_boxes(mask_img: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Get bounding boxes around white regions in binary mask image"""
        mask_array = np.array(mask_img)
        if mask_array.ndim > 2:
            mask_array = mask_array.mean(axis=2)
        mask_array = mask_array > 127

        labeled_array, num_features = label(mask_array)
        logger.info(f"Found {num_features} regions in mask")

        bboxes = []
        for region_id in range(1, num_features + 1):
            region_mask = labeled_array == region_id
            rows = np.any(region_mask, axis=1)
            cols = np.any(region_mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            bboxes.append((x_min, y_min, x_max, y_max))

        return bboxes

    @staticmethod
    def adjust_bbox_region(bbox: tuple) -> tuple:
        """Grow bbox region from center until width or height is 512px or expand by 25% if larger"""
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        if (
            width > InpaintingCropWrapper.MIN_REGION_SIZE
            or height > InpaintingCropWrapper.MIN_REGION_SIZE
        ):
            expansion_factor = 1.25
            new_width = int(width * expansion_factor)
            new_height = int(height * expansion_factor)
        else:
            new_width = max(width, InpaintingCropWrapper.MIN_REGION_SIZE)
            new_height = max(height, InpaintingCropWrapper.MIN_REGION_SIZE)

        new_width = ((new_width + 7) // 8) * 8
        new_height = ((new_height + 7) // 8) * 8

        new_x_min = center_x - new_width // 2
        new_x_max = center_x + new_width // 2
        new_y_min = center_y - new_height // 2
        new_y_max = center_y + new_height // 2

        return (new_x_min, new_y_min, new_x_max, new_y_max)

    @classmethod
    def merge_overlapping_bboxes(cls, bboxes: list) -> list:
        """Merge overlapping bounding boxes after expansion"""
        if not bboxes:
            return []

        expanded_bboxes = [cls.adjust_bbox_region(bbox) for bbox in bboxes]

        while True:
            merged = False
            for i in range(len(expanded_bboxes)):
                for j in range(i + 1, len(expanded_bboxes)):
                    box1 = expanded_bboxes[i]
                    box2 = expanded_bboxes[j]

                    x_min = max(box1[0], box2[0])
                    y_min = max(box1[1], box2[1])
                    x_max = min(box1[2], box2[2])
                    y_max = min(box1[3], box2[3])

                    if x_min < x_max and y_min < y_max:
                        merged_box = (
                            min(box1[0], box2[0]),
                            min(box1[1], box2[1]),
                            max(box1[2], box2[2]),
                            max(box1[3], box2[3]),
                        )
                        expanded_bboxes.pop(j)
                        expanded_bboxes.pop(i)
                        expanded_bboxes.append(merged_box)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break

        return expanded_bboxes

    @staticmethod
    def dilate_mask(
        mask_img: Image.Image, pixels: int = MASK_DILATION_SIZE
    ) -> Image.Image:
        """Dilate the mask by specified number of pixels"""
        mask_array = np.array(mask_img)
        if mask_array.ndim > 2:
            mask_array = mask_array.mean(axis=2)
        mask_array = mask_array > 127

        struct = generate_binary_structure(2, 2)
        for _ in range(pixels - 1):
            struct = binary_dilation(struct)

        dilated_mask = binary_dilation(mask_array, structure=struct)
        dilated_img = Image.fromarray((dilated_mask * 255).astype(np.uint8))

        return dilated_img

    @classmethod
    def extract_bbox_regions(
        cls, img: Image.Image, mask_img: Image.Image, bboxes: list
    ) -> list:
        """Extract regions from both original and mask images based on extended bboxes"""
        merged_bboxes = cls.merge_overlapping_bboxes(bboxes)

        regions = []
        for i, bbox in enumerate(merged_bboxes):
            x_min, y_min, x_max, y_max = bbox

            width, height = img.size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            crop_width = x_max - x_min
            crop_height = y_max - y_min
            crop_width = (crop_width // 8) * 8
            crop_height = (crop_height // 8) * 8
            x_max = x_min + crop_width
            y_max = y_min + crop_height

            if crop_width > 0 and crop_height > 0:
                img_region = img.crop((x_min, y_min, x_max, y_max))
                mask_region = mask_img.crop((x_min, y_min, x_max, y_max))
                mask_region = cls.dilate_mask(mask_region)
                regions.append((img_region, mask_region, (x_min, y_min, x_max, y_max)))

        return regions

    @classmethod
    def create_feathered_mask(
        cls,
        size,
        bbox,
        img_size,
        feather_pixels=BLEND_FEATHER_SIZE,
        edge_threshold=EDGE_PADDING_SIZE,
    ):
        """Create a mask with feathered edges, skipping edges near image boundaries"""
        mask = Image.new("L", size, 255)
        draw = ImageDraw.Draw(mask)

        x_min, y_min, x_max, y_max = bbox
        img_width, img_height = img_size

        feather_left = x_min > edge_threshold
        feather_right = x_max < (img_width - edge_threshold)
        feather_top = y_min > edge_threshold
        feather_bottom = y_max < (img_height - edge_threshold)

        for i in range(feather_pixels):
            alpha = int(255 * (i / feather_pixels))

            if feather_top:
                draw.line([(0, i), (size[0], i)], fill=alpha)
            if feather_bottom:
                draw.line(
                    [(0, size[1] - 1 - i), (size[0], size[1] - 1 - i)], fill=alpha
                )
            if feather_left:
                draw.line([(i, 0), (i, size[1])], fill=alpha)
            if feather_right:
                draw.line(
                    [(size[0] - 1 - i, 0), (size[0] - 1 - i, size[1])], fill=alpha
                )

        return mask

    @classmethod
    def paste_regions(
        cls, original_img: Image.Image, inpainted_regions: list
    ) -> Image.Image:
        """Paste inpainted regions back into original image using alpha mask blending"""
        result_img = original_img.copy().convert("RGBA")
        binary_mask = Image.new("L", original_img.size, 0)

        for i, (inpainted_img, bbox) in enumerate(inpainted_regions):
            x_min, y_min, x_max, y_max = bbox
            region_size = (x_max - x_min, y_max - y_min)

            feathered_mask = cls.create_feathered_mask(
                region_size, bbox, original_img.size
            )
            inpainted_rgba = inpainted_img.convert("RGBA")

            binary_mask.paste(feathered_mask, (x_min, y_min))
            result_img.paste(inpainted_rgba, (x_min, y_min), feathered_mask)

        result_img.paste(original_img, (0, 0), ImageOps.invert(binary_mask))

        return result_img

    def __call__(self, *args, **kwargs):
        try:
            logger.info("Starting inpainting process")

            image = kwargs.get("image")
            mask_image = kwargs.get("mask_image")
            control_image = kwargs.get("control_image")
            num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)

            bboxes = self.get_bounding_boxes(mask_image)
            regions = self.extract_bbox_regions(image, mask_image, bboxes)

            if not regions:
                logger.info("No regions found to inpaint. Returning original image.")

                @dataclass
                class DummyOutput:
                    images: list

                output = DummyOutput(images=[image])
                logger.info("Process completed successfully")
                return output

            inpainted_regions = []
            for i, (img_region, mask_region, bbox) in enumerate(regions):
                logger.info(f"Inpainting region {i+1}/{len(regions)}")

                region_kwargs = kwargs.copy()
                region_kwargs.update(
                    {
                        "image": img_region,
                        "mask_image": mask_region,
                        "height": img_region.size[1],
                        "width": img_region.size[0],
                    }
                )

                if control_image is not None:
                    x_min, y_min, x_max, y_max = bbox
                    control_region = control_image.crop((x_min, y_min, x_max, y_max))
                    region_kwargs["control_image"] = control_region

                output = self._original_call(**region_kwargs)
                logger.info(
                    f"Generated {len(output.images)} images for region {i+1}/{len(regions)}"
                )

                for img in output.images:
                    inpainted_regions.append((img, bbox))

            results = []
            for i in range(num_images_per_prompt):
                regions_for_image = inpainted_regions[i::num_images_per_prompt]
                result = self.paste_regions(image, regions_for_image)
                results.append(result)

            logger.info("Inpainting completed successfully")
            output.images = results
            return output

        except Exception as e:
            logger.error(f"Error during inpainting: {str(e)}")
            raise


# Quick Test
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Input paths
    image_path = Path("assets/interior_2186.png")
    mask_path = Path("assets/interior_2186-mask-multi.png")
    control_image_path = Path("assets/interior_2186-control-seg.png")

    # Load images
    input_image = Image.open(image_path)
    mask_image = Image.open(mask_path)
    control_image = Image.open(control_image_path)

    try:
        # Initialize the base pipeline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_type = torch.float32 if torch.cuda.is_available() else torch.float16

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            INPAINTING_MODEL,
            torch_dtype=weight_type,
        ).to(device)
        controlnet_seg = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_seg",
            torch_dtype=weight_type,
        ).to(device)  # Move to same device

        pipe = StableDiffusionControlNetInpaintPipeline.from_pipe(
            pipe,
            controlnet=controlnet_seg,
        ).to(device)  # Ensure pipeline is on device

        # Wrap the pipeline with cropping functionality
        pipe = InpaintingCropWrapper(pipe)

        # Use the wrapped pipeline
        output = pipe(
            prompt="empty",
            negative_prompt="",
            guidance_scale=7.5,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            controlnet_conditioning_scale=0.6,
            num_images_per_prompt=1,
            num_inference_steps=5,
            control_image=control_image,
            image=input_image,
            mask_image=mask_image,
            strength=1.0,
            sigmas=None
        )

        # Save results
        output_dir = image_path.parent
        base_name = image_path.stem

        for i, result_image in enumerate(output.images):
            output_path = output_dir / f"{base_name}_inpainted_{i}.png"
            result_image.save(output_path)
            logger.info(f"Saved result to {output_path}")

    except Exception as e:
        logger.error(f"Error processing images: {e}")
        sys.exit(1)
