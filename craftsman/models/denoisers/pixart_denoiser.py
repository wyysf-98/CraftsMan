from dataclasses import dataclass

import torch
import torch.nn as nn
import math
import importlib
import craftsman
import re

from typing import Optional
from craftsman.utils.base import BaseModule
from craftsman.models.denoisers.utils import *

@craftsman.register("pixart-denoiser")
class PixArtDinoDenoiser(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None
        input_channels: int = 32
        output_channels: int = 32
        n_ctx: int = 512
        width: int = 768
        layers: int = 28
        heads: int = 16
        context_dim: int = 1024
        n_views: int = 1
        context_ln: bool = True
        init_scale: float = 0.25
        use_checkpoint: bool = False
        drop_path: float = 0.
        clip_weight: float = 1.0
        dino_weight: float = 1.0

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # timestep embedding
        self.time_embed = TimestepEmbedder(self.cfg.width)

        # x embedding
        self.x_embed = nn.Linear(self.cfg.input_channels, self.cfg.width, bias=True)

        # context embedding
        if self.cfg.context_ln:
            self.clip_embed = nn.Sequential(
                nn.LayerNorm(self.cfg.context_dim),
                nn.Linear(self.cfg.context_dim, self.cfg.width),
            )

            self.dino_embed = nn.Sequential(
                nn.LayerNorm(self.cfg.context_dim),
                nn.Linear(self.cfg.context_dim, self.cfg.width),
            )
        else:
            self.clip_embed = nn.Linear(self.cfg.context_dim, self.cfg.width)
            self.dino_embed = nn.Linear(self.cfg.context_dim, self.cfg.width)

        init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        drop_path = [x.item() for x in torch.linspace(0, self.cfg.drop_path, self.cfg.layers)]
        self.blocks = nn.ModuleList([
            DiTBlock(
                    width=self.cfg.width, 
                    heads=self.cfg.heads, 
                    init_scale=init_scale, 
                    qkv_bias=self.cfg.drop_path, 
                    use_flash=True,
                    drop_path=drop_path[i]
            )
            for i in range(self.cfg.layers)
        ])

        self.t_block = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.cfg.width, 6 * self.cfg.width, bias=True)
                    )
        
         # final layer
        self.final_layer = T2IFinalLayer(self.cfg.width, self.cfg.output_channels)

        self.identity_initialize()

        if self.cfg.pretrained_model_name_or_path:
            print(f"Loading pretrained model from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")['state_dict']
            self.denoiser_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith('denoiser_model.'):
                    self.denoiser_ckpt[k.replace('denoiser_model.', '')] = v
            self.load_state_dict(self.denoiser_ckpt, strict=False)

    def identity_initialize(self):
        for block in self.blocks:
            nn.init.constant_(block.attn.c_proj.weight, 0)
            nn.init.constant_(block.attn.c_proj.bias, 0)
            nn.init.constant_(block.cross_attn.c_proj.weight, 0)
            nn.init.constant_(block.cross_attn.c_proj.bias, 0)
            nn.init.constant_(block.mlp.c_proj.weight, 0)
            nn.init.constant_(block.mlp.c_proj.bias, 0)

    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.LongTensor,
                context: torch.FloatTensor):

        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            context (torch.FloatTensor): [bs, context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

        """

        B, n_data, _ = model_input.shape

        # 1. time
        t_emb = self.time_embed(timestep)

        # 2. conditions projector
        context = context.view(B, self.cfg.n_views, -1, self.cfg.context_dim)
        clip_feat, dino_feat = context.chunk(2, dim=2)
        clip_cond = self.clip_embed(clip_feat.contiguous().view(B, -1, self.cfg.context_dim))
        dino_cond = self.dino_embed(dino_feat.contiguous().view(B, -1, self.cfg.context_dim))
        visual_cond = self.cfg.clip_weight * clip_cond + self.cfg.dino_weight * dino_cond

        # 4. denoiser
        latent = self.x_embed(model_input)
        
        t0 = self.t_block(t_emb).unsqueeze(dim=1)
        for block in self.blocks:
            latent = auto_grad_checkpoint(block, latent, visual_cond, t0)

        latent = self.final_layer(latent, t_emb)

        return latent

