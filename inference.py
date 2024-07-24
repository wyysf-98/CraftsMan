import argparse
import os
import sys
import trimesh
import torch
import numpy as np
from glob import glob
from PIL import Image
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from typing import Dict, Tuple, List

from apps.utils import load_model, preprocess_image
from apps.third_party.CRM.pipelines import TwoStagePipeline

model = None
generator = None

sys.path.append(f"apps/third_party/CRM")
crm_pipeline = None

def gen_mvimg(
    image,
    seed,
    guidance_scale,
    step,
    ):
    if seed == 0:
        seed = np.random.randint(1, 65535)

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

def main(input_path:str, outputs_folder:str, weights_folder:str, views:Dict[str, str]=None,
         seed:int=0, device:int=0,
         guidance_scale_2D:float=3, step_2D:int=50,
         guidance_scale_3D:float=3, step_3D:int=50, octree_depth:int=7,
         remesh:bool=False, target_face_count:int=2000,
         ) -> Tuple[str, List[str]]:

    global model, generator, crm_pipeline

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(outputs_folder, exist_ok=True)

    # load the shape diffusion model
    ckpt_path = f"{weights_folder}/model.ckpt"
    config_path = f"{weights_folder}/config.yaml"
    if not model: model = load_model(ckpt_path, config_path, device)
    if not generator: generator = torch.Generator(device)

    # load the multi-view images model
    if not crm_pipeline:
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

    # read the input images
    if os.path.isdir(input_path):
        image_files = glob(os.path.join(input_path, "*.png"))
    else:
        image_files = [input_path]

    for image_file in image_files:
        print(f"Processing {image_file}")
        # generate the multi-view images
        image = Image.open(image_file)
        image = preprocess_image(image, foreground_ratio=1.0)
        mvimages = gen_mvimg(
            image,
            seed,
            guidance_scale_2D,
            step_2D,
        )
        mvimg_paths = []
        for i, mvimg in enumerate(mvimages):
            mvimg_path = os.path.join(outputs_folder, f"{os.path.basename(image_file).split('.')[0]}_{i}.png")
            mvimg.save(mvimg_path)
            mvimg_paths.append(mvimg_path)

        # manually set the view images
        mvimages = list(mvimages)
        views = views or {}
        if front_view_path := views.get("front"):
            view_front = Image.open(front_view_path)
            mvimages[0] = view_front
        if right_view_path := views.get("right"):
            view_right = Image.open(right_view_path)
            mvimages[1] = view_right
        if back_view_path := views.get("back"):
            view_back = Image.open(back_view_path)
            mvimages[2] = view_back
        if left_view_path := views.get("left"):
            view_left = Image.open(left_view_path)
            mvimages[3] = view_left

        # inference the 3D shape
        filepath = os.path.join(outputs_folder, os.path.basename(image_file).split(".")[0] + ".obj")
        out_path = image2mesh(
            np.array(mvimages[0]),
            np.array(mvimages[1]),
            np.array(mvimages[2]),
            np.array(mvimages[3]),
            filepath,
            remesh,
            target_face_count,
            guidance_scale=guidance_scale_3D,
            step=step_3D,
            seed=seed,
            octree_depth=octree_depth
        )
        print(f"Output mesh saved to {out_path}")

    return out_path, mvimg_paths

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./eval_data", help="Path to the input data",)
    parser.add_argument("--output", type=str, default="./eval_outputs", help="Path to the inference results",)
    ############## model and mv model ##############
    parser.add_argument("--model", type=str, default="", help="Path to the image-to-shape diffusion model",)
    ############## inference ##############
    parser.add_argument("--seed", type=int, default=4, help="Random seed for generating multi-view images",)
    parser.add_argument("--guidance_scale_2D", type=float, default=3, help="Guidance scale for generating multi-view images",)
    parser.add_argument("--step_2D", type=int, default=50, help="Number of steps for generating multi-view images",)
    parser.add_argument("--remesh", action="store_true", help="Remesh the output mesh",)
    parser.add_argument("--target_face_count", type=int, default=2000, help="Target face count for remeshing",)
    parser.add_argument("--guidance_scale_3D", type=float, default=3, help="Guidance scale for 3D reconstruction",)
    parser.add_argument("--step_3D", type=int, default=50, help="Number of steps for 3D reconstruction",)
    parser.add_argument("--octree_depth", type=int, default=7, help="Octree depth for 3D reconstruction",)
    ############## data preprocess ##############
    parser.add_argument("--front_view", type=str, default="", help="Front view of the object",)
    parser.add_argument("--right_view", type=str, default="", help="Right view of the object",)
    parser.add_argument("--back_view", type=str, default="", help="Back view of the object",)
    parser.add_argument("--left_view", type=str, default="", help="Left view of the object",)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    views = {}
    if args.front_view != "":
        views["front"] = args.front_view
    if args.right_view != "":
        views["right"] = args.right_view
    if args.back_view != "":
        views["back"] = args.back_view
    if args.left_view != "":
        views["left"] = args.left_view

    main(args.input, args.output, args.model, views=views,
         seed=args.seed, device=args.device,
         guidance_scale_2D=args.guidance_scale_2D, step_2D=args.step_2D,
         guidance_scale_3D=args.guidance_scale_3D, step_3D=args.step_3D, octree_depth=args.octree_depth,
         remesh=args.remesh, target_face_count=args.target_face_count)
 