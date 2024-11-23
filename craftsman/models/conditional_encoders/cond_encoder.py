import random
import torch
from torch import nn
import numpy as np
import re
from einops import rearrange
from dataclasses import dataclass
from torchvision import transforms

from transformers import CLIPTokenizer, CLIPImageProcessor
from transformers import AutoImageProcessor
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer
from transformers.utils import ModelOutput
from typing import Iterable, Optional, Union, List

import craftsman
from craftsman.utils.typing import *
from .clip.modeling_clip import CLIPModel
from .clip.modeling_conditional_clip import ConditionalCLIPModel
from .base import BaseEmbedder, ImageType
from .dino_v2.modeling_dinov2 import Dinov2Model
from .dino_v2.modeling_conditional_dinov2 import ConditionalDinov2Model

@dataclass
class CLIPEmbedOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    embeds: torch.FloatTensor = None

class DINOEmbedOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
        
@craftsman.register("cond-embedder")
class CondEmbedder(BaseEmbedder):

    @dataclass
    class Config(BaseEmbedder.Config):
        pretrained_model_name_or_path: Optional[str] = None # the pretrained model name or path for condition model
        pretrained_clip_name_or_path: Optional[str] = None # the pretrained model name or path for clip
        pretrained_dino_name_or_path: Optional[str] = None # the pretrained model name or path for dino
        pretrained_linear_proj: Optional[str] = None
        freeze_modulation_clip: bool = False
        freeze_modulation_dino: bool = False
        config_path: str = ''
        enable_gradient_checkpointing: bool = False
        embeds_fusion_mode: int = 1 # 0: sum | 1: concat
        linear_proj_init: str = "constant"
        text_max_length: int = 77
        image_size_clip: int = 224
        image_size_dino: int = 224

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # Load the CLIP model and processor
        if not self.cfg.encode_camera:
            if self.cfg.pretrained_clip_name_or_path is not None:
                self.clip_model: CLIPModel = CLIPModel.from_pretrained(self.cfg.pretrained_clip_name_or_path)
            else:
                self.clip_model: CLIPModel = CLIPModel(config=ConditionalCLIPModel.config_class.from_pretrained(
                    "openai/clip-vit-large-patch14",
                ))
            if self.cfg.pretrained_dino_name_or_path is not None:
                self.dino_model: Dinov2Model = Dinov2Model.from_pretrained(self.cfg.pretrained_dino_name_or_path)
            else:
                self.dino_model: Dinov2Model = Dinov2Model(config=ConditionalDinov2Model.config_class.from_pretrained(
                    "facebook/dinov2-base",
                ))
        else:
            if self.cfg.pretrained_clip_name_or_path == '':
                assert self.cfg.config_path is not None, "The config path should be provided"
                conditional_clip_config = ConditionalCLIPModel.config_class.from_json_file(self.cfg.config_path)
                conditional_clip_config.vision_config.modulation_dim = self.cfg.camera_embeds_dim
                self.clip_model: CLIPModel = ConditionalCLIPModel(conditional_clip_config)
            else:
                
                # clip
                conditional_clip_config = ConditionalCLIPModel.config_class.from_pretrained(
                    self.cfg.pretrained_clip_name_or_path,
                )
                conditional_clip_config.vision_config.modulation_dim = self.cfg.camera_embeds_dim
                self.clip_model: CLIPModel = ConditionalCLIPModel.from_pretrained(
                    self.cfg.pretrained_clip_name_or_path, 
                    vision_config=conditional_clip_config.vision_config
                )
                
                # dino
                conditional_vit_config = ConditionalDinov2Model.config_class.from_pretrained(
                    self.cfg.pretrained_dino_name_or_path,
                )                               
                conditional_vit_config.modulation_dim = self.cfg.camera_embeds_dim
                self.dino_model: ConditionalDinov2Model = ConditionalDinov2Model.from_pretrained(
                    self.cfg.pretrained_dino_name_or_path,
                    config=conditional_vit_config
                )
                    
        self.image_preprocess_clip = CLIPImageProcessor()
        self.image_preprocess_dino = AutoImageProcessor.from_pretrained(
            self.cfg.pretrained_dino_name_or_path if self.cfg.pretrained_dino_name_or_path is not None else "facebook/dinov2-base",
        )
        self.transform_clip= transforms.Compose(
            [
                transforms.Resize(self.cfg.image_size_clip, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(self.cfg.image_size_clip),  # crop a (224, 224) square
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        
        self.transform_dino = transforms.Compose(
            [   
                transforms.Resize(self.cfg.image_size_dino, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(self.cfg.image_size_dino),  # crop a (224, 224) square
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if self.cfg.enable_gradient_checkpointing:
            self.dino_model.encoder.gradient_checkpointing = True

        if self.cfg.zero_uncond_embeds:
            self.empty_image_embeds_clip = torch.zeros((self.cfg.n_views, 257, 1024)).detach()
            self.empty_image_embeds_dino = torch.zeros((self.cfg.n_views, 257, 1024)).detach()
            self.empty_image_embeds = torch.cat([self.empty_image_embeds_clip, self.empty_image_embeds_dino], dim=1)
        else:
            if self.cfg.encode_camera:
                self.empty_image_embeds_clip = self.encode_image_clip(torch.zeros(self.cfg.n_views, self.cfg.image_size_clip, self.cfg.image_size_clip, 3), self.cameras[:self.cfg.n_views]).detach()
                self.empty_image_embeds_dino = self.encode_image_dino(torch.zeros(self.cfg.n_views, self.cfg.image_size_clip, self.cfg.image_size_clip, 3), self.cameras[:self.cfg.n_views]).detach()
                self.empty_image_embeds = torch.cat([self.empty_image_embeds_clip, self.empty_image_embeds_dino], dim=1)
            else:
                self.empty_image_embeds_clip = self.encode_image_clip(torch.zeros(self.cfg.n_views, self.cfg.image_size_dino, self.cfg.image_size_dino, 3)).detach()
                self.empty_image_embeds_dino = self.encode_image_dino(torch.zeros(self.cfg.n_views, self.cfg.image_size_dino, self.cfg.image_size_dino, 3)).detach()
                self.empty_image_embeds = torch.cat([self.empty_image_embeds_clip, self.empty_image_embeds_dino], dim=1)

        # Freeze the clip model parameters
        self.clip_model.eval()
        for k, p in self.clip_model.named_parameters():
            ks = k.split('.')
            if 'mod_norm1' in ks or 'mod_norm2' in ks and not self.cfg.freeze_modulation_clip:
                p.requires_grad_(not self.cfg.freeze_modulation_clip)
            else:
                p.requires_grad_(False)
                
        # freeze the dino model parameters
        self.dino_model.eval()
        for k, p in self.dino_model.named_parameters():
            ks = k.split('.')
            if 'mod_norm1' in ks or 'mod_norm2' in ks and not self.cfg.freeze_modulation_dino:
                p.requires_grad_(not self.cfg.freeze_modulation_dino)
            else:
                p.requires_grad_(False)
        
        self.linear_proj = nn.Linear(768, 1024, bias=False)
        if self.cfg.linear_proj_init == "constant":
            nn.init.constant_(self.linear_proj.weight, 0)
        elif self.cfg.linear_proj_init == "xavier":
            nn.init.xavier_uniform_(self.linear_proj.weight)
        else:
            raise ValueError

        if self.cfg.pretrained_model_name_or_path is not None:
            print(f"Loading ckpt from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")['state_dict']
            pretrained_model_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith('condition.'):
                    pretrained_model_ckpt[k.replace('condition.', '')] = v
            self.load_state_dict(pretrained_model_ckpt, strict=False)
        
    def encode_image_clip(self, images: Iterable[Optional[ImageType]], cameras: Optional[torch.Tensor] = None, force_none_camera_embeds: bool = False, return_dict: bool = False, **kwargs) -> torch.FloatTensor:
        camera_embeds = None
        if isinstance(images, (np.ndarray, torch.Tensor)): # for training process
            assert images.min() >= 0.0 and images.max() <= 1.0, "The pixel values should be in the range of [0, 1]"
            do_rescale = False
            if self.cfg.encode_camera:
                assert cameras is not None, "The cameras should be provided"
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.transform_clip(images.permute(0, 3, 1, 2))
        else: # for inference process
            do_rescale = True
            if self.cfg.encode_camera:
                if cameras is None:
                    bs = len(images) // self.cfg.n_views
                    cameras = self.cameras[:self.cfg.n_views].repeat(bs, 1, 1).to(self.clip_model.device)
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.image_preprocess_clip.preprocess(images, return_tensors='pt', do_rescale=do_rescale).pixel_values

        if force_none_camera_embeds:
            camera_embeds = None

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
            if camera_embeds is not None:
                camera_embeds = camera_embeds.unsqueeze(1)
                
        if self.cfg.encode_camera and camera_embeds is not None:
            vision_outputs = self.clip_model.vision_model(
                pixel_values=rearrange(pixel_values.to(self.clip_model.device), "B N C H W -> (B N) C H W"),
                condition=rearrange(camera_embeds, "B N C -> (B N) C")
            )
            
        else:
            vision_outputs = self.clip_model.vision_model(
                pixel_values=rearrange(pixel_values.to(self.clip_model.device), "B N C H W -> (B N) C H W"), 
            )

        if return_dict:
            # clip
            pooler_output = vision_outputs[1]  # pooled_output
            image_features = self.clip_model.visual_projection(pooler_output)

            clip_embeds = vision_outputs.last_hidden_state
            
            clip_embeds_dict = CLIPEmbedOutput(
                last_hidden_state=clip_embeds,
                pooler_output=pooler_output,
                embeds=image_features
            )
            
            return clip_embeds_dict
        else:  
            return vision_outputs.last_hidden_state

    def encode_image_dino(self, images: Iterable[Optional[ImageType]], cameras: Optional[torch.Tensor] = None, force_none_camera_embeds: bool = False, return_dict: bool = False, **kwargs) -> torch.FloatTensor:
        camera_embeds = None
        if isinstance(images, (np.ndarray, torch.Tensor)): # for training process
            assert images.min() >= 0.0 and images.max() <= 1.0, "The pixel values should be in the range of [0, 1]"
            do_rescale = False
            if self.cfg.encode_camera:
                assert cameras is not None, "The cameras should be provided"
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.transform_dino(images.permute(0, 3, 1, 2))
        else: # for inference process
            do_rescale = True
            if self.cfg.encode_camera:
                if cameras is None:
                    bs = len(images) // self.cfg.n_views
                    cameras = self.cameras[:self.cfg.n_views].repeat(bs, 1, 1).to(self.dino_model.device)
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.image_preprocess_dino.preprocess(images, return_tensors='pt', do_rescale=do_rescale).pixel_values

        if force_none_camera_embeds:
            camera_embeds = None

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
            if camera_embeds is not None:
                camera_embeds = camera_embeds.unsqueeze(1)

        if self.cfg.encode_camera and camera_embeds is not None:
            vision_outputs = self.dino_model(
                rearrange(pixel_values.to(self.dino_model.device), "B N C H W -> (B N) C H W"), 
                condition=rearrange(camera_embeds, "B N C -> (B N) C"),
            )
        else:
            
            vision_outputs = self.dino_model(
                rearrange(pixel_values.to(self.dino_model.device), "B N C H W -> (B N) C H W"), 
            )

        if return_dict:
            # dino
            dino_embeds_dict = DINOEmbedOutput(
                last_hidden_state=vision_outputs.last_hidden_state,
                pooler_output=vision_outputs.pooler_output,
            )
            return dino_embeds_dict
        else:
            return vision_outputs.last_hidden_state

    def encode_image(self, images: Iterable[Optional[ImageType]], cameras: Optional[torch.Tensor] = None, force_none_camera_embeds: bool = False, return_dict: bool = False, **kwargs) -> torch.FloatTensor:
        clip_embeds = self.encode_image_clip(images, cameras)
        dino_embeds = self.encode_image_dino(images, cameras)
        dino_embeds = self.linear_proj(dino_embeds)
        visual_embeds = torch.cat([clip_embeds, dino_embeds], dim=1)
        return visual_embeds
