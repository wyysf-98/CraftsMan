from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir))
import time
import cv2
import gradio as gr
import numpy as np
import torch
import PIL
from PIL import Image
import rembg
from rembg import remove
rembg_session = rembg.new_session()
from segment_anything import sam_model_registry, SamPredictor

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.config import ExperimentConfig, load_config

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")
    
def load_model(
        ckpt_path: str, 
        config_path: str, 
        device = "cuda"
    ):
    cfg: ExperimentConfig
    cfg = load_config(config_path)
    
    if 'pretrained_model_name_or_path' not in cfg.system.condition_model or cfg.system.condition_model.pretrained_model_name_or_path is None:
        cfg.system.condition_model.config_path = config_path.replace("config.yaml", "clip_config.json")

    system: BaseSystem = craftsman.find(cfg.system_type)(
        cfg.system, 
    )
    
    print(f"Restoring states from the checkpoint path at {ckpt_path} with config {cfg}")
    system.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    system = system.to(device).eval()

    return system

class RMBG(object):
    def __init__(self, device):
        sam = sam_model_registry["vit_h"](checkpoint=f"{parent_dir}/ckpts/SAM/sam_vit_h_4b8939.pth").to(device)
        self.predictor = SamPredictor(sam)
        
    def rmbg_sam(self, input_image):
        def _sam_segment(predictor, input_image, *bbox_coords):
            bbox = np.array(bbox_coords)
            image = np.asarray(input_image)

            start_time = time.time()
            predictor.set_image(image)

            masks_bbox, scores_bbox, logits_bbox = predictor.predict(box=bbox, multimask_output=True)

            print(f"SAM Time: {time.time() - start_time:.3f}s")
            out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            out_image[:, :, :3] = image
            out_image_bbox = out_image.copy()
            out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
            torch.cuda.empty_cache()
            return Image.fromarray(out_image_bbox, mode='RGBA')

        RES = 1024
        input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)

        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:, :, -1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        return _sam_segment(self.predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
        
    def rmbg_rembg(self, input_image):
        def _rembg_remove(
            image: PIL.Image.Image,
            rembg_session = None,
            force: bool = False,
            **rembg_kwargs,
        ) -> PIL.Image.Image:
            do_remove = True
            if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
                # explain why current do not rm bg
                print("alhpa channl not enpty, skip remove background, using alpha channel as mask")
                background = Image.new("RGBA", image.size, (0, 0, 0, 0))
                image = Image.alpha_composite(background, image)
                do_remove = False
            do_remove = do_remove or force
            if do_remove:
                image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
            return image
        return _rembg_remove(input_image, rembg_session, force_remove=True)

    def run(self, rm_type, image, foreground_ratio, background_choice, backgroud_color):
        # image = cv2.resize(np.array(image), (crop_size, crop_size))
        # image = Image.fromarray(image)

        if background_choice == "Alpha as mask":
            image = do_resize_content(image, foreground_ratio)
            image = expand_to_square(image)
            image = add_background(image, backgroud_color)
            return image
        
        elif "Remove" in background_choice:
            if rm_type.upper() == "SAM":
                image = self.rmbg_sam(image)
            elif rm_type.upper() == "REMBG":
                image = self.rmbg_rembg(image)
            else:
                return -1

            image = do_resize_content(image, foreground_ratio)
            image = expand_to_square(image)
            # image = add_background(image, backgroud_color)
            return image
    
        elif "Original" in background_choice:
            return image
        else:
            return -1
        
def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image
    
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def add_background(image, bg_color=(255, 255, 255, 0)):
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)