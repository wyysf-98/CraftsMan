import gradio as gr
import numpy as np
import torch
import PIL
from PIL import Image
import os
import sys
import rembg
import time
import json
import cv2
from datetime import datetime
from einops import repeat, rearrange
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from .utils import *
from huggingface_hub import hf_hub_download

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GenMVImage(object):
    def __init__(self, device):
        self.seed = 1024
        self.guidance_scale = 7.5
        self.step = 50
        self.pipelines = {}
        self.device = device
        
    def gen_image_from_crm(self, image):
        
        from third_party.CRM.pipelines import TwoStagePipeline
        specs = json.load(open(f"{parent_dir}/apps/third_party/CRM/configs/specs_objaverse_total.json"))
        stage1_config = OmegaConf.load(f"{parent_dir}/apps/third_party/CRM/configs/nf7_v3_SNR_rd_size_stroke.yaml").config
        stage1_sampler_config = stage1_config.sampler
        stage1_model_config = stage1_config.models
        stage1_model_config.resume = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth", repo_type="model")
        stage1_model_config.config = f"{parent_dir}/apps/third_party/CRM/" + stage1_model_config.config
        if "crm" in self.pipelines.keys():
            pipeline = self.pipelines['crm']
        else:
            self.pipelines['crm'] = TwoStagePipeline(
                                        stage1_model_config,
                                        stage1_sampler_config,
                                        device=self.device,
                                        dtype=torch.float16
                                    )
            pipeline = self.pipelines['crm']
        pipeline.set_seed(self.seed)
        rt_dict = pipeline(image, scale=self.guidance_scale, step=self.step)
        mv_imgs = rt_dict["stage1_images"]
        return mv_imgs[5], mv_imgs[3], mv_imgs[2], mv_imgs[0]
    
    def gen_image_from_mvdream(self, image, text):
        from third_party.mvdream_diffusers.pipeline_mvdream import MVDreamPipeline
        if image is None:
            if "mvdream" in self.pipelines.keys():
                pipe_MVDream = self.pipelines['mvdream']
            else:
                self.pipelines['mvdream'] = MVDreamPipeline.from_pretrained(
                    "ashawkey/mvdream-sd2.1-diffusers", # remote weights
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                self.pipelines['mvdream'] = self.pipelines['mvdream'].to(self.device)
                pipe_MVDream = self.pipelines['mvdream']
            mv_imgs = pipe_MVDream(
                    text,
                    negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
                    num_inference_steps=self.step,
                    guidance_scale=self.guidance_scale,
                    generator = torch.Generator(self.device).manual_seed(self.seed)
                )
        else:
            image = np.array(image)
            image = image.astype(np.float32) / 255.0
            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
            if "imagedream" in self.pipelines.keys():
                pipe_imagedream = self.pipelines['imagedream']
            else:
                self.pipelines['imagedream'] = MVDreamPipeline.from_pretrained(
                        "ashawkey/imagedream-ipmv-diffusers", # remote weights
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    )
                self.pipelines['imagedream'] = self.pipelines['imagedream'].to(self.device)
                pipe_imagedream = self.pipelines['imagedream']
            mv_imgs = pipe_imagedream(
                        text,
                        image,
                        negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
                        num_inference_steps=self.step,
                        guidance_scale=self.guidance_scale,
                        generator = torch.Generator(self.device).manual_seed(self.seed)
                    )
        return mv_imgs[1], mv_imgs[2], mv_imgs[3], mv_imgs[0]
    
    def gen_image_from_wonder3d(self, image, crop_size):
        sys.path.append(f"{parent_dir}/apps/third_party/Wonder3D")
        from third_party.Wonder3D.mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
        from third_party.Wonder3D.utils.misc import load_config
        weight_dtype = torch.float16
        batch = prepare_data(image, crop_size)

        if "wonder3d" in self.pipelines.keys():
            pipeline = self.pipelines['wonder3d']
        else:
            self.pipelines['wonder3d'] = MVDiffusionImagePipeline.from_pretrained(
                        'flamehaze1115/wonder3d-v1.0',
                        custom_pipeline=f'{parent_dir}/apps/third_party/Wonder3D/mvdiffusion/pipelines/pipeline_mvdiffusion_image.py',
                        torch_dtype=weight_dtype
                    )
            self.pipelines['wonder3d'].unet.enable_xformers_memory_efficient_attention()
            self.pipelines['wonder3d'].to(self.device)
            self.pipelines['wonder3d'].set_progress_bar_config(disable=True)
            pipeline = self.pipelines['wonder3d']
            
        generator = torch.Generator(device=pipeline.unet.device).manual_seed(self.seed)
        # repeat  (2B, Nv, 3, H, W)
        imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0).to(weight_dtype)

        # (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0).to(weight_dtype)

        task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0).to(weight_dtype)

        camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(weight_dtype)

        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")
        # (B*Nv, Nce)

        out = pipeline(
            imgs_in,
            # camera_embeddings,
            generator=generator,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.step,
            output_type='pt',
            num_images_per_prompt=1,
            **{'eta': 1.0},
        ).images

        bsz = out.shape[0] // 2
        normals_pred = out[:bsz]
        images_pred = out[bsz:]

        normals_pred = [save_image(normals_pred[i]) for i in range(bsz)]
        images_pred = [save_image(images_pred[i]) for i in range(bsz)]

        mv_imgs = images_pred
        return mv_imgs[0], mv_imgs[2], mv_imgs[4], mv_imgs[5]
        
    def run(self, mvimg_model, text, image, crop_size, seed, guidance_scale, step):
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.step = step
        if mvimg_model.upper() == "CRM":
            return self.gen_image_from_crm(image)
        elif mvimg_model.upper() == "IMAGEDREAM":
            return self.gen_image_from_mvdream(image, text)
        elif mvimg_model.upper() == "WONDER3D":
            return self.gen_image_from_wonder3d(image, crop_size)