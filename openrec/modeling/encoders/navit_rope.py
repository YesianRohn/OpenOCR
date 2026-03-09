from __future__ import annotations

from functools import partial
from typing import List
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence



def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0


class GELU(nn.Module):

    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):

    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'hard_sigmoid':
            self.act = nn.Hardsigmoid(inplace)
        elif act_type == 'hard_swish':
            self.act = nn.Hardswish(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=inplace)
        elif act_type == 'gelu':
            self.act = GELU(inplace=inplace)
        elif act_type == 'swish':
            self.act = Swish(inplace=inplace)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)


def drop_path(x,
              drop_prob: float = 0.0,
              training: bool = False,
              scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RoPEAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis, attn_mask=None):
        B, N, C = x.shape
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn += attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, freqs_cis, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis, attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    # 2D rotary embedding
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class Embed(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 bias=False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_chans),
            nn.Linear(in_chans, embed_dim, bias=bias),
            nn.LayerNorm(embed_dim),
        )


    def forward(self, x):
        x = self.proj(x)
        return x


class NaViT_ROPE(nn.Module):
    def __init__(
        self,
        patch_size=[4, 8],
        in_channels=3,
        out_channels=256,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        last_stage=False,
        feat2d=False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = embed_dim
        self.patch_size = patch_size
        self.token_dropout_rate = 0.1
        patch_dim = in_channels * (patch_size[0] * patch_size[1])
        self.patch_embed = Embed(patch_dim, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=act_layer,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

        self.rope_theta = 100.0
        self.num_heads = num_heads

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_images: List[torch.Tensor]):
        p, device = self.patch_size, self.device
        arange = partial(torch.arange, device=device)
        pad_sequence = partial(orig_pad_sequence, batch_first=True) 

        # process images into variable lengthed sequences with attention mask
        sequences = []
        freqs_cis_list = []

        for image in batched_images:
            h, w = image.shape[-2:]
            ph, pw = h // p[1], w // p[0]
            pos = torch.stack(torch.meshgrid((
                arange(ph),
                arange(pw)
            ), indexing='ij'), dim=-1)
            pos = pos.view(ph * pw, -1)
            seq = image.view(-1, ph, p[1], pw, p[0]).permute(1, 3, 0, 2, 4).reshape(ph * pw, -1)
            seq_len = seq.shape[-2]

            if self.training and self.token_dropout_rate > 0:
                num_keep = max(1, int(seq_len * (1 - self.token_dropout_rate)))
                keep_indices = torch.randperm(seq_len, device=device)[:num_keep]
                seq = seq[keep_indices]
                pos = pos[keep_indices]

            # freq_cis
            fc = compute_axial_cis(
                dim=self.embed_dim // self.num_heads,
                end_x=pw,
                end_y=ph,
                theta=self.rope_theta
            )  # (ph*pw, head_dim)
            idx = pos[:, 0] * pw + pos[:, 1]
            fc = fc[idx.to(fc.device)]  # (N, head_dim)

            sequences.append(seq)
            freqs_cis_list.append(fc)

        # derive key padding mask
        lengths = torch.tensor([seq.shape[-2] for seq in sequences], device=device, dtype=torch.long)
        seq_arange = torch.arange(lengths.amax().item(), device=device)
        mask = seq_arange.unsqueeze(0) >= lengths.unsqueeze(1)  # [batch, seq]
        attn_mask = torch.where(mask, float('-inf'), 0.0)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        x = pad_sequence(sequences).to(device)
        freqs_cis = pad_sequence(freqs_cis_list).to(device)

        x = self.patch_embed(x)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, freqs_cis, attn_mask)

        x = self.norm(x)

        return x, mask

