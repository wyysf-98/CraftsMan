import argparse
import os
import sys
import json
import trimesh
import torch
import numpy as np
from glob import glob
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict
from huggingface_hub import hf_hub_download

from apps.utils import load_model, RMBG
from apps.third_party.LGM.pipeline_mvdream import MVDreamPipeline
from apps.third_party.CRM.pipelines import TwoStagePipeline
from apps.third_party.Wonder3D.mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

model = None
generator = None

sys.path.append(f"apps/third_party/CRM")
crm_pipeline = None

sys.path.append(f"apps/third_party/LGM")
imgaedream_pipeline = None

sys.path.append(f"apps/third_party/Wonder3D")
wonder3d_pipeline = None

def gen_mvimg(
    mvimg_model, 
    image, 
    seed, 
    guidance_scale, 
    step, 
    text, 
    neg_text, 
    elevation, 
    backgroud_color
):
    if seed == 0:
        seed = np.random.randint(1, 65535)

    if mvimg_model == "CRM":
        global crm_pipeline
        crm_pipeline.set_seed(seed)
        background = Image.new("RGBA", image.size, (127, 127, 127)) # force background to be gray 
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

def image2mesh(view_front: np.ndarray, 
               view_right: np.ndarray, 
               view_back: np.ndarray, 
               view_left: np.ndarray,
               filepath: str,
               remesh: bool = False,
               target_face_count: int = 2000,
               scheluder_name: str ="DDIMScheduler",
               guidance_scale: int = 7.5,
               step: int = 50,
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
        steps=step,
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
    mesh.export(filepath, include_normals=True)

    if remesh:
        remeshed_filepath = filepath.replace(".obj", "_remeshed.obj")
        print("Remeshing with Instant Meshes...")
        command = f"apps/third_party/InstantMeshes {filepath} -f {target_face_count} -o {remeshed_filepath}"
        os.system(command)
        filepath = remeshed_filepath
    
    return filepath

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./eval_data", help="Path to the input data",)
    parser.add_argument("--output", type=str, default="./eval_outputs", help="Path to the inference results",)
    ############## model and mv model ##############
    parser.add_argument("--model", type=str, default="", help="Path to the image-to-shape diffusion model",)
    parser.add_argument("--mv_model", type=str, default="CRM", help="Path to the multi-view images model",)
    ############## inference ##############
    parser.add_argument("--seed", type=int, default=4, help="Random seed for generating multi-view images",)
    parser.add_argument("--guidance_scale_2D", type=float, default=3, help="Guidance scale for generating multi-view images",)
    parser.add_argument("--step_2D", type=int, default=50, help="Number of steps for generating multi-view images",)
    parser.add_argument("--remesh", type=bool, default=False, help="Remesh the output mesh",)
    parser.add_argument("--target_face_count", type=int, default=2000, help="Target face count for remeshing",)
    parser.add_argument("--guidance_scale_3D", type=float, default=3, help="Guidance scale for 3D reconstruction",)
    parser.add_argument("--step_3D", type=int, default=50, help="Number of steps for 3D reconstruction",)
    parser.add_argument("--octree_depth", type=int, default=7, help="Octree depth for 3D reconstruction",)
    ############## data preprocess ##############
    parser.add_argument("--no_rmbg", type=bool, default=False, help="Do NOT remove the background",)
    parser.add_argument("--rm_type", type=str, default="rembg", choices=["rembg", "sam"], help="Type of background removal",)
    parser.add_argument("--bkgd_type", type=str, default="Remove", choices=["Alpha as mask", "Remove", "Original"], help="Type of background",)
    parser.add_argument("--bkgd_color", type=str, default="[127,127,127,255]", help="Background color",)
    parser.add_argument("--fg_ratio", type=float, default=1.0, help="Foreground ratio",)
    parser.add_argument("--front_view", type=str, default="", help="Front view of the object",)
    parser.add_argument("--right_view", type=str, default="", help="Right view of the object",)
    parser.add_argument("--back_view", type=str, default="", help="Back view of the object",)
    parser.add_argument("--left_view", type=str, default="", help="Left view of the object",)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    
    # load the shape diffusion model
    if args.model == "":
        ckpt_path = hf_hub_download(repo_id="wyysf/CraftsMan", filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/model.ckpt", repo_type="model")
        config_path = hf_hub_download(repo_id="wyysf/CraftsMan", filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/config.yaml", repo_type="model")
    else:
        ckpt_path = f"{args.model}/model.ckpt"
        config_path = f"{args.model}/config.yaml"    
    model = load_model(ckpt_path, config_path, device)
    generator = torch.Generator(device)

    # load the multi-view images model
    if args.mv_model == "CRM":
        stage1_config = OmegaConf.load(f"apps/third_party/CRM/configs/nf7_v3_SNR_rd_size_stroke.yaml").config
        stage1_sampler_config = stage1_config.sampler
        stage1_model_config = stage1_config.models
        stage1_model_config.resume = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth", repo_type="model")
        stage1_model_config.config = f"apps/third_party/CRM/" + stage1_model_config.config
        crm_pipeline = TwoStagePipeline(
                            stage1_model_config,
                            stage1_sampler_config,
                            device=device,
                            dtype=torch.float16
                        )
    elif args.mv_model == "ImageDream":
        imagedream_pipeline = MVDreamPipeline.from_pretrained(
            "ashawkey/imagedream-ipmv-diffusers", # remote weights
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        imagedream_pipeline.to(device)
    elif args.mv_model == "Wonder3D":
        wonder3d_pipeline = DiffusionPipeline.from_pretrained(
            'flamehaze1115/wonder3d-v1.0', # remote weights
            custom_pipeline='flamehaze1115/wonder3d-pipeline',
            torch_dtype=torch.float16
        )
        wonder3d_pipeline.unet.enable_xformers_memory_efficient_attention()
        wonder3d_pipeline.to(device)
    else:
        raise ValueError(f"Unsupported multi-view images model: {args.mv_model}")
    
    # read the input images
    if os.path.isdir(args.input):
        image_files = glob(os.path.join(args.input, "*.png"))
    else:
        image_files = [args.input]
    
    if args.no_rmbg:
        rmbg = None
    else:
        # remove the background
        rmbg = RMBG(device)
    
    for image_file in image_files:
        print(f"Processing {image_file}")
        # generate the multi-view images
        image = Image.open(image_file)
        if not args.no_rmbg:
            image = rmbg.run(args.rm_type, image, args.fg_ratio, args.bkgd_type, args.bkgd_color)
        mvimages = gen_mvimg(
            args.mv_model, 
            image, 
            args.seed, 
            args.guidance_scale_2D, 
            args.step_2D, 
            "", # text for ImageDream
            "ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate",
            0.0, 
            json.loads(args.bkgd_color)
        )
        for i, mvimg in enumerate(mvimages):
            mvimg.save(os.path.join(args.output, f"{os.path.basename(image_file).split('.')[0]}_{i}.png"))
            
        # manually set the view images
        if args.front_view != "":
            view_front = Image.open(args.front_view)
            mvimages[0] = view_front
        if args.right_view != "":
            view_right = Image.open(args.right_view)
            mvimages[1] = view_right
        if args.back_view != "":
            view_back = Image.open(args.back_view)
            mvimages[2] = view_back
        if args.left_view != "":
            view_left = Image.open(args.left_view)
            mvimages[3] = view_left
        
        # inference the 3D shape    
        filepath = os.path.join(args.output, os.path.basename(image_file).split(".")[0] + ".obj")
        out_path = image2mesh(
            np.array(mvimages[0]), 
            np.array(mvimages[1]), 
            np.array(mvimages[2]), 
            np.array(mvimages[3]),
            filepath,
            args.remesh,
            args.target_face_count,
            guidance_scale=args.guidance_scale_3D,
            step=args.step_3D,
            seed=args.seed,
            octree_depth=args.octree_depth
        )
        print(f"Output mesh saved to {out_path}")