import torch
from libs.base_utils import do_resize_content
from imagedream.ldm.util import (
    instantiate_from_config,
    get_obj_from_str,
)
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from inference import generate3d
from huggingface_hub import hf_hub_download
import json
import argparse
import shutil
from model import CRM
import PIL
import rembg
import os
from pipelines import TwoStagePipeline

rembg_session = rembg.new_session()

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


def preprocess_image(image, background_choice, foreground_ratio, backgroud_color):
    """
    input image is a pil image in RGBA, return RGB image
    """
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputdir",
        type=str,
        default="examples/kunkun.webp",
        help="dir for input image",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--bg_choice",
        type=str,
        default="Auto Remove background",
        help="[Auto Remove background] or [Alpha as mask]",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="out/",
    )    
    args = parser.parse_args()
    

    img = Image.open(args.inputdir)
    img = preprocess_image(img, args.bg_choice, 1.0, (127, 127, 127))
    os.makedirs(args.outdir, exist_ok=True)
    img.save(args.outdir+"preprocessed_image.png")

    crm_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")
    specs = json.load(open("configs/specs_objaverse_total.json"))
    model = CRM(specs).to("cuda")
    model.load_state_dict(torch.load(crm_path, map_location = "cuda"), strict=False)

    stage1_config = OmegaConf.load("configs/nf7_v3_SNR_rd_size_stroke.yaml").config
    stage2_config = OmegaConf.load("configs/stage2-v2-snr.yaml").config
    stage2_sampler_config = stage2_config.sampler
    stage1_sampler_config = stage1_config.sampler

    stage1_model_config = stage1_config.models
    stage2_model_config = stage2_config.models

    xyz_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="ccm-diffusion.pth")
    pixel_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth")
    stage1_model_config.resume = pixel_path
    stage2_model_config.resume = xyz_path

    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
    )

    rt_dict = pipeline(img, scale=args.scale, step=args.step)
    stage1_images = rt_dict["stage1_images"]
    stage2_images = rt_dict["stage2_images"]
    np_imgs = np.concatenate(stage1_images, 1)
    np_xyzs = np.concatenate(stage2_images, 1)
    Image.fromarray(np_imgs).save(args.outdir+"pixel_images.png")
    Image.fromarray(np_xyzs).save(args.outdir+"xyz_images.png")

    glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, "cuda")
    shutil.copy(obj_path, args.outdir+"output3d.zip")