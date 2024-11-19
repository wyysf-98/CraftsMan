import spaces
import argparse
import os
import json
import torch
import sys
import time
import importlib
import numpy as np
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline

import PIL
from PIL import Image
from collections import OrderedDict
import trimesh
import rembg
import gradio as gr
from typing import Any

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir))

import tempfile

import craftsman
from craftsman.utils.config import ExperimentConfig, load_config

_TITLE = '''CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner'''
_DESCRIPTION = '''
<div>
<span style="color: red;">Important: If you have your own data and want to collaborate, we are welcom to any contact.</span>
<div>
Select or upload a image, then just click 'Generate'. 
<br>
By mimicking the artist/craftsman modeling workflow, we propose CraftsMan (aka 匠心) that uses 3D Latent Set Diffusion Model that directly generate coarse meshes,
then a multi-view normal enhanced image generation model is used to refine the mesh.
We provide the coarse 3D diffusion part here. 
<br>
If you found CraftsMan is helpful, please help to ⭐ the <a href='https://github.com/wyysf-98/CraftsMan/' target='_blank'>Github Repo</a>. Thanks!
<a style="display:inline-block; margin-left: .5em" href='https://github.com/wyysf-98/CraftsMan/'><img src='https://img.shields.io/github/stars/wyysf-98/CraftsMan?style=social' /></a>
<br>
*If you have your own multi-view images, you can directly upload it.
</div>
'''
_CITE_ = r"""
---
📝 **Citation**
If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{li2024craftsman,
author    = {Weiyu Li and Jiarui Liu and Rui Chen and Yixun Liang and Xuelin Chen and Ping Tan and Xiaoxiao Long},
title     = {CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner},
journal   = {arXiv preprint arXiv:2405.14979},
year      = {2024},
}
```
🤗 **Acknowledgements**
We use <a href='https://github.com/wjakob/instant-meshes' target='_blank'>Instant Meshes</a> to remesh the generated mesh to a lower face count, thanks to the authors for the great work.
📋 **License**
CraftsMan is under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html), so any downstream solution and products (including cloud services) that include CraftsMan code or a trained model (both pretrained or custom trained) inside it should be open-sourced to comply with the AGPL conditions. If you have any questions about the usage of CraftsMan, please contact us first.
📧 **Contact**
If you have any questions, feel free to open a discussion or contact us at <b>weiyuli.cn@gmail.com</b>.
"""

model = None
cached_dir = None

generator = None

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

class RMBG(object):
    def __init__(self):
        pass

    def rmbg_rembg(self, input_image, background_color):
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
                background = Image.new("RGBA", image.size, background_color)
                image = Image.alpha_composite(background, image)
                do_remove = False
            do_remove = do_remove or force
            if do_remove:
                image = rembg.remove(image, session=rembg_session, **rembg_kwargs)

            # calculate the min bbox of the image
            alpha = image.split()[-1]
            image = image.crop(alpha.getbbox())

            return image
        return _rembg_remove(input_image, None, force_remove=True)

    def run(self, rm_type, image, foreground_ratio, background_choice, background_color=(0, 0, 0, 0)):
        if "Original" in background_choice:
            return image
        else:
            if background_choice == "Alpha as mask":
                alpha = image.split()[-1]
                image = image.crop(alpha.getbbox())
            
            elif "Remove" in background_choice:
                if rm_type.upper() == "REMBG":
                    image = self.rmbg_rembg(image, background_color=background_color)
                else:
                    return -1
        
            # Calculate the new size after rescaling
            new_size = tuple(int(dim * foreground_ratio) for dim in image.size)
            # Resize the image while maintaining the aspect ratio
            resized_image = image.resize(new_size)
            # Create a new image with the original size and white background
            padded_image = PIL.Image.new("RGBA", image.size, (0, 0, 0, 0))
            paste_position = ((image.width - resized_image.width) // 2, (image.height - resized_image.height) // 2)
            padded_image.paste(resized_image, paste_position)

            # expand image to 1:1
            width, height = padded_image.size
            if width == height:
                return padded_image
            new_size = (max(width, height), max(width, height))
            image = PIL.Image.new("RGBA", new_size, (0, 0, 0, 0))
            paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
            image.paste(padded_image, paste_position)
            return image

# @spaces.GPU
def image2mesh(image: Any, 
               more: bool = False,
               scheluder_name: str ="DDIMScheduler",
               guidance_scale: int = 7.5,
               steps: int = 30,
               seed: int = 4,
               target_face_count: int = 2000,
               octree_depth: int = 7):
    
    sample_inputs = {
        "image": [
            image
        ]
    }

    global model
    latents = model.sample(
        sample_inputs,
        sample_times=1,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed
    )[0]
    
    # decode the latents to mesh
    box_v = 1.1
    mesh_outputs, _ = model.shape_model.extract_geometry(
        latents,
        bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
        octree_depth=octree_depth
    )
    assert len(mesh_outputs) == 1, "Only support single mesh output for gradio demo"
    mesh = trimesh.Trimesh(mesh_outputs[0][0], mesh_outputs[0][1])
    # filepath = f"{cached_dir}/{time.time()}.obj"
    filepath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
    mesh.export(filepath, include_normals=True)

    if 'Remesh' in more:
        remeshed_filepath = tempfile.NamedTemporaryFile(suffix=f"_remeshed.obj", delete=False).name
        print("Remeshing with Instant Meshes...")
        command = f"{proj_dir}/apps/third_party/InstantMeshes {filepath} -f {target_face_count} -o {remeshed_filepath}"
        os.system(command)
        filepath = remeshed_filepath
    
    return filepath

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./ckpts/craftsman-v1-5", help="Path to the object file",)
    parser.add_argument("--cached_dir", type=str, default="")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()


    cached_dir = args.cached_dir
    if cached_dir != "":
        os.makedirs(args.cached_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # for input image
    background_choice = OrderedDict({ 
        "Alpha as Mask": "Alpha as Mask",
        "Auto Remove Background": "Auto Remove Background",
        "Original Image": "Original Image",        
    })

    generator = torch.Generator(device)

    # for 3D latent set diffusion
    if args.model_path == "":
        ckpt_path = hf_hub_download(repo_id="craftsman3d/craftsman-v1-5", filename="model.ckpt", repo_type="model")
        config_path = hf_hub_download(repo_id="craftsman3d/craftsman-v1-5", filename="config.yaml", repo_type="model")
    else:
        ckpt_path = os.path.join(args.model_path, "model.ckpt")
        config_path = os.path.join(args.model_path, "config.yaml")
    scheluder_dict = OrderedDict({ 
        "DDIMScheduler": 'diffusers.schedulers.DDIMScheduler',
        # "DPMSolverMultistepScheduler": 'diffusers.schedulers.DPMSolverMultistepScheduler', # not support yet
        # "UniPCMultistepScheduler": 'diffusers.schedulers.UniPCMultistepScheduler', # not support yet
    })
    
    # main GUI
    custom_theme = gr.themes.Soft(primary_hue="blue").set(
                    button_secondary_background_fill="*neutral_100",
                    button_secondary_background_fill_hover="*neutral_200")
    custom_css = '''#disp_image {
        text-align: center; /* Horizontally center the content */
    }'''
    
    with gr.Blocks(title=_TITLE, theme=custom_theme, css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Column():
                    # input image
                    with gr.Row():
                        image_input = gr.Image(
                            label="Image Input",
                            image_mode="RGBA",
                            sources="upload",
                            type="pil",
                        )
                run_btn = gr.Button('Generate', variant='primary', interactive=True)

                with gr.Row():
                    gr.Markdown('''Try a different <b>seed and MV Model</b> for better results. Good Luck :)''')
                with gr.Row():
                    seed = gr.Number(0, label='Seed', show_label=True)
                    more = gr.CheckboxGroup(["Remesh"], label="More", show_label=False)
                    target_face_count = gr.Number(2000, label='Target Face Count', show_label=True)

                with gr.Row():
                    gr.Examples(
                        examples=[os.path.join("./asset/examples", i) for i in os.listdir("./asset/examples")],
                        inputs=[image_input],
                        examples_per_page=8
                    )

            with gr.Column(scale=4):
                with gr.Row():
                    output_model_obj = gr.Model3D(
                        label="Output Model (OBJ Format)",
                        camera_position=(90.0, 90.0, 3.5),
                        interactive=False,
                    )
                with gr.Row():
                    gr.Markdown('''*please note that the model is fliped due to the gradio viewer, please download the obj file and you will get the correct orientation.''')
                    
                with gr.Accordion('Advanced options', open=False):
                    with gr.Row():
                        background_choice = gr.Dropdown(label="Backgroud Choice", value="Auto Remove Background",choices=list(background_choice.keys()))
                        rmbg_type = gr.Dropdown(label="Backgroud Remove Type", value="rembg",choices=['sam', "rembg"])
                        foreground_ratio = gr.Slider(label="Foreground Ratio", value=1.0, minimum=0.5, maximum=1.0, step=0.01)

                    with gr.Row():
                        guidance_scale = gr.Number(label="3D Guidance Scale", value=5.0, minimum=3.0, maximum=10.0)
                        steps = gr.Number(value=50, minimum=20, maximum=100, label="3D Sample Steps")
                        
                    with gr.Row():
                        scheduler = gr.Dropdown(label="scheluder", value="DDIMScheduler",choices=list(scheluder_dict.keys()))
                        octree_depth = gr.Slider(label="Octree Depth", value=7, minimum=4, maximum=8, step=1)
                    
        gr.Markdown(_CITE_)

        outputs = [output_model_obj]
        rmbg = RMBG()
        
        # model = load_model(ckpt_path, config_path, device)
        cfg = load_config(config_path)
        model = craftsman.find(cfg.system_type)(cfg.system)
        print(f"Restoring states from the checkpoint path at {ckpt_path} with config {cfg}")
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(
            ckpt["state_dict"] if "state_dict" in ckpt else ckpt,
        )
        model = model.to(device).eval()


        run_btn.click(fn=check_input_image, inputs=[image_input]
                    ).success(
                            fn=rmbg.run, 
                            inputs=[rmbg_type, image_input, foreground_ratio, background_choice],
                            outputs=[image_input]
                    ).success(
                            fn=image2mesh, 
                            inputs=[image_input, more, scheduler, guidance_scale, steps, seed, target_face_count, octree_depth],
                            outputs=outputs, 
                            api_name="generate_img2obj")
        
        demo.queue().launch(share=True, allowed_paths=[args.cached_dir])
