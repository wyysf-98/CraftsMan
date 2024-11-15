import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from itertools import repeat
from collections.abc import Iterable
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from timm.models.layers import DropPath
from craftsman.models.transformers.utils import MLP
from craftsman.models.transformers.attention import MultiheadAttention, MultiheadCrossAttention

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class DiTBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )

        self.norm2 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)

    def forward(self, x, visual_cond, t, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, visual_cond)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x

class DiTBlock_text(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )

        self.cross_attn_extra = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )
        self.norm2 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)

    def forward(self, x, visual_cond, text_cond, t, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, visual_cond)
        x = x + self.cross_attn_extra(x, text_cond)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )
        self.norm2 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)

    def forward(self, x, y, t, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x
    
def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

# def t2i_modulate(x, shift, scale):
#     a = torch.ones_like(scale)
#     a[..., 768:] = 0
#     return x * (a + scale) + shift

def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, 'grad_checkpointing', False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, **kwargs)
    return module(*args, **kwargs)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype
    

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, pe_interpolation=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / pe_interpolation
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed