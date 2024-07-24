import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir))
import torch
from PIL import Image

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.config import ExperimentConfig, load_config

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def preprocess_image(image:Image, foreground_ratio:float):
    image = do_resize_content(image, foreground_ratio)
    image = expand_to_square(image)
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
