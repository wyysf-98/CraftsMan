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

from collections import OrderedDict
import trimesh
import gradio as gr
from typing import Any

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir))

import tempfile

from apps.utils import *

_TITLE = '''CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner'''
_DESCRIPTION = '''
<div>
<span style="color: red;">Important: The ckpt models released have been primarily trained on character data, hence they are likely to exhibit superior performance in this category. We are also planning to release more advanced pretrained models in the future.</span>
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
from apps.third_party.LGM.pipeline_mvdream import MVDreamPipeline
from apps.third_party.CRM.pipelines import TwoStagePipeline
from apps.third_party.Wonder3D.mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline


model = None
cached_dir = None
stage1_config = OmegaConf.load(f"apps/third_party/CRM/configs/nf7_v3_SNR_rd_size_stroke.yaml").config
stage1_sampler_config = stage1_config.sampler
stage1_model_config = stage1_config.models
stage1_model_config.resume = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth", repo_type="model")
stage1_model_config.config = f"apps/third_party/CRM/" + stage1_model_config.config
crm_pipeline = None

sys.path.append(f"apps/third_party/LGM")
imgaedream_pipeline = None

sys.path.append(f"apps/third_party/Wonder3D")
wonder3d_pipeline = None

generator = None

@spaces.GPU
def gen_mvimg(
    mvimg_model, image, seed, guidance_scale, step, text, neg_text, elevation, backgroud_color
):
    if seed == 0:
        seed = np.random.randint(1, 65535)

    if mvimg_model == "CRM":
        global crm_pipeline
        crm_pipeline.set_seed(seed)
        background = Image.new("RGBA", image.size, (127, 127, 127))
        image = Image.alpha_composite(background, image)
        mv_imgs = crm_pipeline(
            image, 
            scale=guidance_scale, 
            step=step
        )["stage1_images"]
        return mv_imgs[5], mv_imgs[3], mv_imgs[2], mv_imgs[0]
    
    elif mvimg_model == "ImageDream":
        global imagedream_pipeline, generator
        background = Image.new("RGBA", image.size, backgroud_color)
        image = Image.alpha_composite(background, image)

        image = np.array(image).astype(np.float32) / 255.0
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        mv_imgs = imagedream_pipeline(
            text, 
            image, 
            negative_prompt=neg_text, 
            guidance_scale=guidance_scale,  
            num_inference_steps=step, 
            elevation=elevation,
            generator=generator.manual_seed(seed),
        )
        return mv_imgs[1], mv_imgs[2], mv_imgs[3], mv_imgs[0]
    
    elif mvimg_model == "Wonder3D":
        global wonder3d_pipeline
        background = Image.new("RGBA", image.size, backgroud_color)
        image = Image.alpha_composite(background, image)

        image = Image.fromarray(np.array(image).astype(np.uint8)[..., :3]).resize((256, 256))
        mv_imgs = wonder3d_pipeline(
            image, 
            guidance_scale=guidance_scale, 
            num_inference_steps=step,
            generator=generator.manual_seed(seed),
        ).images
        return mv_imgs[0], mv_imgs[2], mv_imgs[4], mv_imgs[5]

@spaces.GPU
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
    # filepath = f"{cached_dir}/{time.time()}.obj"
    filepath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
    mesh.export(filepath, include_normals=True)

    if 'Remesh' in more:
        remeshed_filepath = tempfile.NamedTemporaryFile(suffix=f"_remeshed.obj", delete=False).name
        print("Remeshing with Instant Meshes...")
        # target_face_count = int(len(mesh.faces)/10)
        target_face_count = 2000
        command = f"{proj_dir}/apps/third_party/InstantMeshes {filepath} -f {target_face_count} -o {remeshed_filepath}"
        os.system(command)
        filepath = remeshed_filepath
        # filepath = filepath.replace('.obj', '_remeshed.obj')
    
    return filepath

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="Path to the object file",)
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
    crm_pipeline = TwoStagePipeline(
                        stage1_model_config,
                        stage1_sampler_config,
                        device=device,
                        dtype=torch.float16
                    )
    imagedream_pipeline = MVDreamPipeline.from_pretrained(
        "ashawkey/imagedream-ipmv-diffusers", # remote weights
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    imagedream_pipeline.to(device)
    wonder3d_pipeline = DiffusionPipeline.from_pretrained(
        'flamehaze1115/wonder3d-v1.0', # remote weights
        custom_pipeline='flamehaze1115/wonder3d-pipeline',
        torch_dtype=torch.float16
    )
    # enable xformers
    wonder3d_pipeline.unet.enable_xformers_memory_efficient_attention()
    wonder3d_pipeline.to(device)

    generator = torch.Generator(device)


    # for 3D latent set diffusion
    if args.model_path == "":
        ckpt_path = hf_hub_download(repo_id="wyysf/CraftsMan", filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/model.ckpt", repo_type="model")
        config_path = hf_hub_download(repo_id="wyysf/CraftsMan", filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/config.yaml", repo_type="model")
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
                    mvimg_model = gr.Dropdown(value="CRM", label="MV Image Model", choices=list(mvimg_model_config_list))
                    # mvimg_model = gr.Dropdown(value="Wonder3D", label="MV Image Model", choices=list(mvimg_model_config_list))
                    more = gr.CheckboxGroup(["Remesh", "Symmetry(TBD)"], label="More", show_label=False)
                with gr.Row():
                    # input prompt
                    text = gr.Textbox(label="Prompt (Opt.)", info="only works for ImageDream")

                with gr.Accordion('Advanced options', open=False):
                    # negative prompt
                    neg_text = gr.Textbox(label="Negative Prompt", value='ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate')
                    # elevation
                    elevation = gr.Slider(label="elevation", minimum=-90, maximum=90, step=1, value=0)

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
                    gr.Markdown('''*please note that the model is fliped due to the gradio viewer, please download the obj file and you will get the correct orientation.''')
                    
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
                        # backgroud_color = gr.ColorPicker(label="Background Color", value="#7F7F7F", interactive=True)
                        
                    with gr.Row():
                        mvimg_guidance_scale = gr.Number(value=3.0, minimum=3, maximum=10, label="2D Guidance Scale")
                        mvimg_steps = gr.Number(value=30, minimum=20, maximum=100, label="2D Sample Steps")
            
                with gr.Accordion('Advanced options (3D)', open=False):
                    with gr.Row():
                        guidance_scale = gr.Number(label="3D Guidance Scale", value=5.0, minimum=3.0, maximum=10.0)
                        steps = gr.Number(value=50, minimum=20, maximum=100, label="3D Sample Steps")
                        
                    with gr.Row():
                        scheduler = gr.Dropdown(label="scheluder", value="DDIMScheduler",choices=list(scheluder_dict.keys()))
                        octree_depth = gr.Slider(label="Octree Depth", value=7, minimum=4, maximum=8, step=1)
                    
        gr.Markdown(_CITE_)

        outputs = [output_model_obj]
        rmbg = RMBG(device)
        
        model = load_model(ckpt_path, config_path, device)

        run_btn.click(fn=check_input_image, inputs=[image_input]
                    ).success(
                            fn=rmbg.run, 
                            inputs=[rmbg_type, image_input, foreground_ratio, background_choice, backgroud_color],
                            outputs=[image_input]
                    ).success(
                            fn=gen_mvimg,
                            inputs=[mvimg_model, image_input, seed, mvimg_guidance_scale, mvimg_steps, text, neg_text, elevation, backgroud_color],
                            outputs=[view_front, view_right, view_back, view_left]
                    ).success(
                            fn=image2mesh, 
                            inputs=[view_front, view_right, view_back, view_left, more, scheduler, guidance_scale, seed, octree_depth],
                            outputs=outputs, 
                            api_name="generate_img2obj")
        run_mv_btn.click(fn=gen_mvimg,
                        inputs=[mvimg_model, image_input, seed, mvimg_guidance_scale, mvimg_steps, text, neg_text, elevation, backgroud_color],
                        outputs=[view_front, view_right, view_back, view_left]
        )
        run_3d_btn.click(fn=image2mesh, 
                        inputs=[view_front, view_right, view_back, view_left, more, scheduler, guidance_scale, seed, octree_depth],
                        outputs=outputs, 
                        api_name="generate_img2obj")
        
        demo.queue().launch(share=True, allowed_paths=[args.cached_dir])
