import os
import warnings
from typing import Callable, List, Optional, Union, Dict, Any
import PIL.Image
import trimesh
import rembg
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from diffusers.utils import BaseOutput

import craftsman
from craftsman.utils.config import ExperimentConfig, load_config

class MeshPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[trimesh.Trimesh]` or `np.ndarray`)
            List of denoised trimesh meshes of length `batch_size` or a tuple of NumPy array with shape `((vertices, 3), (faces, 3)) of length `batch_size``.
    """

    meshes: Union[List[trimesh.Trimesh], np.ndarray]


class CraftsManPipeline():
    """
    Pipeline for text-guided image to image generation using CraftsMan(https://github.com/wyysf-98/CraftsMan).

    Args:
        feature_extractor ([`CLIPFeatureExtractor`]):
            Feature extractor for image pre-processing before being encoded.
    """
    def __init__(
        self,
        device: str,
        cfg: ExperimentConfig,
        system,
    ):
        self.device = device
        self.cfg = cfg
        self.system = system

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        A simpler version that instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.
        The pipeline is set in evaluation mode (`model.eval()`) by default.
        """
        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            ckpt_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="model.ckpt", repo_type="model")
            config_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="config.yaml", repo_type="model")
        else:
            ckpt_path = os.path.join(pretrained_model_name_or_path, "model.ckpt")
            config_path = os.path.join(pretrained_model_name_or_path, "config.yaml")

        # 2. Load the model
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        cfg = load_config(config_path)
        system = craftsman.find(cfg.system_type)(cfg.system)
        print(f"Restoring states from the checkpoint path at {ckpt_path} with config {cfg}")
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        system.load_state_dict(
            ckpt["state_dict"] if "state_dict" in ckpt else ckpt,
        )
        system = system.to(device).eval()

        return cls(
            device=device,
            cfg=cfg,
            system=system
        )

    def check_inputs(
        self,
        image,
    ):
        r"""
        Check if the inputs are valid. Raise an error if not.
        """
        if isinstance(image, str):
            assert os.path.isfile(image) or image.startswith("http"), "Input image must be a valid URL or a file path."
        elif not isinstance(image, (torch.Tensor, PIL.Image.Image)):
            raise ValueError("Input image must be a `torch.Tensor` or `PIL.Image.Image`.")
        
    def preprocess_image(
        self,
        images_pil: List[PIL.Image.Image],
        force: bool = False,
        background_color: List[int] = [255, 255, 255],
        foreground_ratio: float = 1.0,
    ):
        r"""
        Crop and remote the background of the input image
        Args:
            image_pil (`List[PIL.Image.Image]`):
                List of `PIL.Image.Image` objects representing the input image.
            force (`bool`, *optional*, defaults to `False`):
                Whether to force remove the background even if the image has an alpha channel.
        Returns:
            `List[PIL.Image.Image]`: List of `PIL.Image.Image` objects representing the preprocessed image.
        """
        preprocessed_images = []
        for i in range(len(images_pil)):
            image = images_pil[i]
            do_remove = True
            if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
                # explain why current do not rm bg
                print("alhpa channl not enpty, skip remove background, using alpha channel as mask")
                background = PIL.Image.new("RGBA", image.size, (*background_color, 0))
                image = PIL.Image.alpha_composite(background, image)
                do_remove = False
            do_remove = do_remove or force
            if do_remove:
                image = rembg.remove(image)

            # calculate the min bbox of the image
            alpha = image.split()[-1]
            image = image.crop(alpha.getbbox())

            # Calculate the new size after rescaling
            new_size = tuple(int(dim * foreground_ratio) for dim in image.size)
            # Resize the image while maintaining the aspect ratio
            resized_image = image.resize(new_size)
            # Create a new image with the original size and white background
            padded_image = PIL.Image.new("RGBA", image.size, (*background_color, 0))
            paste_position = ((image.width - resized_image.width) // 2, (image.height - resized_image.height) // 2)
            padded_image.paste(resized_image, paste_position)

            # expand image to 1:1
            width, height = padded_image.size
            if width == height:
                preprocessed_images.append(padded_image)
                continue
            new_size = (max(width, height), max(width, height))
            new_image = PIL.Image.new("RGBA", new_size, (*background_color, 1))
            paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
            new_image.paste(padded_image, paste_position)
            preprocessed_images.append(new_image)

        return preprocessed_images

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        num_meshes_per_prompt: Optional[int] = 1,
        output_type: Optional[str] = "trimesh",
        return_dict: bool = True,
        seed: Optional[int] = None,
        force_remove_background: bool = False,
        background_color: List[int] = [255, 255, 255],
        foreground_ratio: float = 0.95,
        mc_depth: int = 8,
        only_max_component: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch. The image will be encoded to its CLIP/DINO-v2 embedding 
                which the DiT will be conditioned on. 
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter as defined in [DDIM](https://arxiv.org/abs/2010.02502). `eta` is a parameter that
                controls the amount of noise added to the latent space. It is only used with the DDIM scheduler and
                will be ignored for other schedulers. `eta` should be between [0, 1].
            num_meshes_per_prompt (`int`, *optional*, defaults to 1):
                The number of meshes to generate per prompt.
            output_type (`str`, *optional*, defaults to `"trimesh"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image`, `latents` or `np.array of v and f`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            seed (`int`, *optional*, defaults to `None`):
                Seed for the random number generator. Setting a seed will ensure reproducibility.
            force_remove_background (`bool`, *optional*, defaults to `False`):
                Whether to force remove the background even if the image has an alpha channel.
            foreground_ratio (`float`, *optional*, defaults to 1.0):
                The ratio of the foreground in the image. The foreground is the part of the image that is not the
                background. The foreground is resized to the size of the background image while maintaining the aspect
                ratio. The background is filled with black color. The foreground ratio should be between [0, 1].
            mc_depth (`int`, *optional*, defaults to 8):
                The resolution of the Marching Cubes algorithm. The resolution is the number of cubes in the x, y, and z.
                8 means 2^8 = 256 cubes in each dimension. The higher the resolution, the more detailed the mesh will be.
            only_max_component (`bool`, *optional*, defaults to `False`):
                Whether to only keep the largest connected component of the mesh. This is useful when the mesh has
                multiple components and only the largest one is needed.
        Examples:

        Returns:
            [`~MeshPipelineOutput`] or `tuple`: [`~MeshPipelineOutput`] if `return_dict` is True, otherwise a `tuple`. 
            When returning a tuple, the first element is a list with the generated meshes.
        """
        # 0. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
        )

        # 1. Define call parameters
        if isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        elif isinstance(image, PIL.Image.Image) or isinstance(image, str):
            batch_size = 1
        do_classifier_free_guidance = guidance_scale != 1.0

        # 2. Preprocess input image
        if isinstance(image, torch.Tensor):
            images_pil = [TF.to_pil_image(image[i]) for i in range(image.shape[0])]
        elif isinstance(image, PIL.Image.Image):
            images_pil = [image]
        elif isinstance(image, str):
            if image.startswith("http"):
                import requests
                images_pil = [PIL.Image.open(requests.get(image, stream=True).raw)]
            else:
                images_pil = [PIL.Image.open(image)]
        images_pil = self.preprocess_image(
            images_pil, 
            force=force_remove_background,
            background_color=background_color,
            foreground_ratio=foreground_ratio
            )

        # 3. Inference 
        latents = self.system.sample(
            {'image': images_pil},
            sample_times = num_meshes_per_prompt,
            steps = num_inference_steps,
            guidance_scale = guidance_scale,
            eta = eta,
            seed = seed
        )

        # 4. Post-processing
        if not output_type == "latent":
            mesh = []
            for i, cur_latents in enumerate(latents):
                print(f"Generating mesh {i+1}/{num_meshes_per_prompt}")
                mesh_v_f, has_surface = self.system.shape_model.extract_geometry(
                    cur_latents, 
                    octree_depth=mc_depth,
                    extract_mesh_func="mc"
                )
                
                if output_type == "trimesh":
                    import trimesh
                    cur_mesh = trimesh.Trimesh(vertices=mesh_v_f[0][0], faces=mesh_v_f[0][1])
                    if only_max_component:
                        components = cur_mesh.split(only_watertight=False)
                        bbox = []
                        for c in components:
                            bbmin = c.vertices.min(0)
                            bbmax = c.vertices.max(0)
                            bbox.append((bbmax - bbmin).max())
                        max_component = np.argmax(bbox)
                        cur_mesh = components[max_component]
                    mesh.append(cur_mesh)
                elif output_type == "np":
                    mesh.append(mesh_v_f[0])
        else:
            mesh = latents

        if not return_dict:
            return tuple(mesh)
        return MeshPipelineOutput(meshes=mesh)