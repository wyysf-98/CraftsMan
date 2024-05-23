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

from collections import OrderedDict
import trimesh
from einops import repeat, rearrange
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple, List
import gradio as gr
from utils import *

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir))

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.config import ExperimentConfig, load_config

from mv_models import GenMVImage

_TITLE = '''CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner'''
_DESCRIPTION = '''
<div>
Select or upload a image, then just click 'Generate'. 
<br>
By mimicking the artist/craftsman modeling workflow, we propose CraftsMan (aka 匠心) that uses 3D Latent Set Diffusion Model that directly generate coarse meshes,
then a multi-view normal enhanced image generation model is used to refine the mesh.
We provide the coarse 3D diffusion part here. 
<br>
If you found Crafts is helpful, please help to ⭐ the <a href='https://github.com/wyysf-98/CraftsMan/' target='_blank'>Github Repo</a>. Thanks!
<a style="display:inline-block; margin-left: .5em" href='https://github.com/wyysf-98/CraftsMan/'><img src='https://img.shields.io/github/stars/wyysf-98/CraftsMan?style=social' /></a>
<br>
*please note that the model is fliped due to the gradio viewer, please download the obj file and you will get the correct mesh.
<br>
*If you have your own multi-view images, you can directly upload it.
</div>
'''
_CITE_ = r"""
---
📝 **Citation**
If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{craftsman,
author    = {Weiyu Li and Jiarui Liu and Rui Chen and Yixun Liang and Xuelin Chen and Ping Tan and Xiaoxiao Long},
title     = {CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner},
journal   = {arxiv:xxx},
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

def image2mesh(view_front: np.ndarray, 
               view_right: np.ndarray, 
               view_back: np.ndarray, 
               view_left: np.ndarray,
               more: bool = False,
               scheluder_name: str ="DDIMScheduler",
               guidance_scale: int = 7.5,
               seed: int = 4,
               octree_depth: int = 7):
    
    sample_inputs = {
        "mvimages": [[
            Image.fromarray(view_front), 
            Image.fromarray(view_right), 
            Image.fromarray(view_back), 
            Image.fromarray(view_left)
        ]]
    }

    global model
    latents = model.sample(
        sample_inputs,
        sample_times=1,
        guidance_scale=guidance_scale,
        return_intermediates=False,
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
    filepath = f"{cached_dir}/{time.time()}.obj"
    mesh.export(filepath, include_normals=True)

    if 'Remesh' in more:
        print("Remeshing with Instant Meshes...")
        target_face_count = int(len(mesh.faces)/10)
        command = f"{proj_dir}/apps/third_party/InstantMeshes {filepath} -f {target_face_count} -d -S 0 -r 6 -p 6 -o {filepath.replace('.obj', '_remeshed.obj')}"
        os.system(command)
        filepath = filepath.replace('.obj', '_remeshed.obj')
    
    return filepath
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, required=True, help="Path to the object file",)
    parser.add_argument("--cached_dir", type=str, default="./gradio_cached_dir")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cached_dir = args.cached_dir
    os.makedirs(args.cached_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # for multi-view images generation
    background_choice = OrderedDict({ 
        "Alpha as Mask": "Alpha as Mask",
        "Auto Remove Background": "Auto Remove Background",
        "Original Image": "Original Image",        
    })
    mvimg_model_config_list = ["CRM", "ImageDream", "Wonder3D"]
    
    # for 3D latent set diffusion
    # for 3D latent set diffusion
    ckpt_path = hf_hub_download(repo_id="wyysf/CraftsMan", filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6/model.ckpt", repo_type="model")
    config_path = hf_hub_download(repo_id="wyysf/CraftsMan", filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6/config.yaml", repo_type="model")
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
                with gr.Row():
                    image_input = gr.Image(
                        label="Image Input",
                        image_mode="RGBA",
                        sources="upload",
                        type="pil",
                    )
                with gr.Row():
                    text = gr.Textbox(label="Prompt (Optional, only works for mvdream)", visible=False)
                with gr.Row():
                    gr.Markdown('''Try a different <b>seed</b> if the result is unsatisfying. Good Luck :)''')
                with gr.Row():
                    seed = gr.Number(42, label='Seed', show_label=True)
                    more = gr.CheckboxGroup(["Remesh", "Symmetry(TBD)"], label="More", show_label=False)
                    # remesh = gr.Checkbox(value=False, label='Remesh')
                    # symmetry = gr.Checkbox(value=False, label='Symmetry(TBD)', interactive=False)
                run_btn = gr.Button('Generate', variant='primary', interactive=True)
                
                with gr.Row():
                    gr.Examples(
                        examples=[os.path.join("./apps/examples", i) for i in os.listdir("./apps/examples")],
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
                    view_front = gr.Image(label="Front", interactive=True, show_label=True)
                    view_right = gr.Image(label="Right", interactive=True, show_label=True)
                    view_back = gr.Image(label="Back", interactive=True, show_label=True)
                    view_left = gr.Image(label="Left", interactive=True, show_label=True)
                    
                with gr.Accordion('Advanced options', open=False):
                    with gr.Row(equal_height=True):
                        run_mv_btn = gr.Button('Only Generate 2D', interactive=True)
                        run_3d_btn = gr.Button('Only Generate 3D', interactive=True)

                with gr.Accordion('Advanced options (2D)', open=False):
                    with gr.Row():
                        crop_size = gr.Number(224, label='Crop size')
                        mvimg_model = gr.Dropdown(value="CRM", label="MV Image Model", choices=mvimg_model_config_list)
                
                    with gr.Row():
                        foreground_ratio = gr.Slider(
                                label="Foreground Ratio",
                                minimum=0.5,
                                maximum=1.0,
                                value=1.0,
                                step=0.05,
                            )
                        
                    with gr.Row():
                        background_choice = gr.Dropdown(label="Backgroud Choice", value="Auto Remove Background",choices=list(background_choice.keys()))
                        rmbg_type = gr.Dropdown(label="Backgroud Remove Type", value="rembg",choices=['sam', "rembg"])
                        backgroud_color = gr.ColorPicker(label="Background Color", value="#FFFFFF", interactive=True)
                        
                    with gr.Row():
                        mvimg_guidance_scale = gr.Number(value=3.5, minimum=3, maximum=10, label="2D Guidance Scale")
                        mvimg_steps = gr.Number(value=50, minimum=20, maximum=100, label="2D Sample Steps", precision=0)
            
                with gr.Accordion('Advanced options (3D)', open=False):
                    with gr.Row():
                        guidance_scale = gr.Number(label="3D Guidance Scale", value=7.5, minimum=3.0, maximum=10.0)
                        steps = gr.Number(value=50, minimum=20, maximum=100, label="3D Sample Steps", precision=0)
                        
                    with gr.Row():
                        scheduler = gr.Dropdown(label="scheluder", value="DDIMScheduler",choices=list(scheluder_dict.keys()))
                        octree_depth = gr.Slider(label="Octree Depth", value=7, minimum=4, maximum=8, step=1)
                    
        gr.Markdown(_CITE_)

        outputs = [output_model_obj]
        rmbg = RMBG(device)
        
        gen_mvimg = GenMVImage(device)
        model = load_model(ckpt_path, config_path, device)

        run_btn.click(fn=check_input_image, inputs=[image_input]
                    ).success(
                            fn=rmbg.run, 
                            inputs=[rmbg_type, image_input, crop_size, foreground_ratio, background_choice, backgroud_color],
                            outputs=[image_input]
                    ).success(
                            fn=gen_mvimg.run,
                            inputs=[mvimg_model, text, image_input, crop_size, seed, mvimg_guidance_scale, mvimg_steps],
                            outputs=[view_front, view_right, view_back, view_left]
                    ).success(
                            fn=image2mesh, 
                            inputs=[view_front, view_right, view_back, view_left, more, scheduler, guidance_scale, seed, octree_depth],
                            outputs=outputs, 
                            api_name="generate_img2obj")
        run_mv_btn.click(fn=gen_mvimg.run,
                        inputs=[mvimg_model, text, image_input, crop_size, seed, mvimg_guidance_scale, mvimg_steps],
                        outputs=[view_front, view_right, view_back, view_left]
        )
        run_3d_btn.click(fn=image2mesh, 
                        inputs=[view_front, view_right, view_back, view_left, more, scheduler, guidance_scale, seed, octree_depth],
                        outputs=outputs, 
                        api_name="generate_img2obj")
        
        demo.queue().launch(share=True, allowed_paths=[args.cached_dir])