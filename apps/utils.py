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
from craftsman.utils.config import ExperimentConfig, load_config

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(
        ckpt_path: str, 
        config_path: str, 
        scheluder_name: str = None,
        scheluder_dict : dict = None,
        device = "cuda"
    ):
    cfg: ExperimentConfig
    cfg = load_config(config_path)
    
    if 'pretrained_model_name_or_path' not in cfg.system.condition_model or cfg.system.condition_model.pretrained_model_name_or_path is None:
        cfg.system.condition_model.config_path = config_path.replace("config.yaml", "clip_config.json")

    # cfg.system.denoise_scheduler= {
    #     'num_train_timesteps': 1000,
    #     'beta_start': 0.00085,
    #     'beta_end': 0.012,
    #     'beta_schedule': 'scaled_linear',
    #     'steps_offset': 1
    # }

    system: BaseSystem = craftsman.find(cfg.system_type)(
        cfg.system, 
    )
    
    print(f"Restoring states from the checkpoint path at {ckpt_path} with config {cfg}")
    system.load_state_dict(torch.load(ckpt_path)['state_dict'])
    system = system.to(device).eval()

    return system

def rmbg_sam(iamge, foreground_ratio):
    return iamge

def rmbg_rembg(iamge, foreground_ratio):
    return iamge

class RMBG(object):
    def __init__(self, device):
        sam_checkpoint = f"{parent_dir}/ckpts/SAM/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        self.predictor = SamPredictor(sam)
        
    def rmbg_sam(self, input_image, crop_size, foreground_ratio, segment=True, rescale=True):
        RES = 1024
        input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)

        if segment:
            image_rem = input_image.convert('RGBA')
            image_nobg = remove(image_rem, alpha_matting=True)
            arr = np.asarray(image_nobg)[:, :, -1]
            x_nonzero = np.nonzero(arr.sum(axis=0))
            y_nonzero = np.nonzero(arr.sum(axis=1))
            x_min = int(x_nonzero[0].min())
            y_min = int(y_nonzero[0].min())
            x_max = int(x_nonzero[0].max())
            y_max = int(y_nonzero[0].max())
            input_image = sam_segment(self.predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
        # Rescale and recenter
        if rescale:
            image_arr = np.array(input_image)
            in_w, in_h = image_arr.shape[:2]
            out_res = min(RES, max(in_w, in_h))
            ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
            x, y, w, h = cv2.boundingRect(mask)
            max_size = max(w, h)
            side_len = int(max_size / foreground_ratio)
            padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
            center = side_len // 2
            padded_image[center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w] = image_arr[y : y + h, x : x + w]
            rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)

            rgba_arr = np.array(rgba) / 255.0
            rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
            input_image = Image.fromarray((rgb * 255).astype(np.uint8))
        else:
            input_image = expand2square(input_image, (127, 127, 127, 0))
        return input_image

        
    def rmbg_rembg(self, image, crop_size, foreground_ratio, background_choice, backgroud_color):
        print(background_choice)
        if background_choice == "Alpha as mask":
            background = Image.new("RGBA", image.size, (0, 0, 0, 0))
            image = Image.alpha_composite(background, image)
        else:
            image = remove_background(image, rembg_session, force_remove=True)
        image = do_resize_content(image, foreground_ratio)
        image = expand_to_square(image)
        image = add_background(image, backgroud_color)
        return image.convert("RGB")

    def run(self, rm_type, image, crop_size, foreground_ratio, background_choice, backgroud_color):
        if "Remove" in background_choice:
            if rm_type.upper() == "SAM":
                return self.rmbg_sam(image, crop_size, foreground_ratio, background_choice, backgroud_color)
            elif rm_type.upper() == "REMBG":
                return self.rmbg_rembg(image, crop_size, foreground_ratio, background_choice, backgroud_color)
            else:
                return -1
        elif "Original" in background_choice:
            return image
        else:
            return -1
        

def save_image(tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    return ndarr

def prepare_data(single_image, crop_size):
    from third_party.Wonder3D.mvdiffusion.data.single_image_dataset import SingleImageDataset
    dataset = SingleImageDataset(root_dir='', num_views=6, img_wh=[256, 256], bg_color='white', crop_size=crop_size, single_image=single_image)
    return dataset[0]

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


def sam_segment(predictor, input_image, *bbox_coords):
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

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

def remove_background(
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

def add_background(image, bg_color=(255, 255, 255)):
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)