#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-12 23:01:21
Author: joh
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange



class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.conv = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1
        )
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.activation = nn.SiLU()

        self.in_channels = in_channels
    
    def forward(self, x, film_modulation = None):
        # B C H W
        # film = gamma, beta for modulation
        assert x.shape[1] == self.in_channels, f"Missmatch channel : got {x.shape}, expected : {self.in_channels}"
        out = self.conv(x)
        out = self.norm(out)

        if film_modulation:
            gamma, beta = film_modulation
            # gamma size b, out_channels, 1, 1 -> broadcast 
            # +1 bc normal init? 
            out = out * (gamma + 1) + beta
            
        out = self.activation(out)

        return out



class ResBlock(nn.Module):
    # B1, B2, res connection
    def __init__(self, in_channels, out_channels, groups, time_emb_dim):
        super().__init__()
        self.b1 = Block(in_channels=in_channels, out_channels=out_channels, groups=groups)
        self.b2 = Block(in_channels=out_channels, out_channels=out_channels, groups=groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels=in_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=1, 
                                        padding=0,
                                        stride=1
            )
        self.in_channels = in_channels
        self.out_channels = out_channels

        if time_emb_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, 2*out_channels)
            )
        
        
        self.time_emb_dim = time_emb_dim

    def forward(self, x, time_emb = None):
        B, C, H, W = x.shape
        film_modulation = None
        if time_emb is not None and self.mlp is not None:
            assert time_emb.shape[1] == self.time_emb_dim, f"miss match shape mlp and time: {time_emb.shape} and {self.time_emb_dim}"
            film_features = self.mlp(time_emb)
            # b, 2*out_channels 
            # needs features for each pixel

            film_features = rearrange(film_features, 'b c -> b c 1 1') # avec c = 2*hidden_dim
            film_modulation = film_features.chunk(2, dim=1) # tuple, size of each component b hidden dim 1 1


        out = self.b1(x, film_modulation)
        out = self.b2(out)

        if self.in_channels != self.out_channels:
            x = self.channel_up(x)

        return out + x


class Attend(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # compuite qkT -> softmax -> dot v
        # q : [B, num_heads, seq_len, hidden_dim]

        # usual writting: q @ k avec k on transpose (B, N, hiddden_dim, seq_len) + make contiguous  
        # and simply do @
    
        # try with einsum notation 
        # a @ b = torch.einsum(“ij,jk->ik”, A, B)
        # the sum is compute along the missing dim

        scale = q.shape[-1] ** -0.5

        # b, h, i, j et ici i=j
        qkt = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale

        s = F.softmax(qkt, dim=-1)
        s = self.attn_dropout(s)

        # o = s @ v
        o = torch.einsum("b h i j, b h j d -> b h i d", s, v)

        return o

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, norm_fn=RMSNorm):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = norm_fn(dim)
        self.attention = Attend()
        # same as doing nn.Linear with Wq Wk Wv, bc kernel_size=1 so no spatial mixing and no bias
        # -> b, hidden_dim * 3, h, w
        self.to_qkv = nn.Conv2d(
                        in_channels=dim,
                        out_channels=hidden_dim*3,
                        kernel_size=1,
                        bias=False
        )

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x)

        # (b, hidden_dim * 3, h, w) -> tuple of 3 tenssors of size (b, hidden_dim, h, w)
        qkv = qkv.chunk(3, dim=1)

        # split hidden dim to heads
        # b hidden_dim, h,w -> b num_heads, h*w, hidden_dim/num_heads
        # x = h, y=w, h = num_heads, c=hiddendim
        # (h c) means c = k such as h*k = c_old
        q, k, v = map(lambda x: rearrange(x, "b (h c) x y -> b h (x y) c", h = self.heads), qkv)
        out = self.attention(q, k, v)
        # b, h, xy, c
        # re mettre en bonne forme
        # b h (x y) c -> b (hc) x y 
        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=h, y=w)
        # seq len remap to 2D and alll heads 'merges'
        
        # out proj
        return self.to_out(out)


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Original formula from "Attention is all you need": PE(pos, i) = func(pos / 10000^(i/dim))
        # The following positional encodings are evaluated in log space.
        # Input: (B). Output: (B, dim)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


#need to check why using that way
def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )


# same
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

class AttenionFiLMResUNet(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels, 
            initial_channels,
            base_channels,
            dim_mults = (1, 2, 4, 8),
            groups=8,
    ):
        super().__init__()

        self.in_proj = nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=initial_channels,
                            kernel_size=7,
                            padding=3
        )

        # premier modulation apres le premier block du premier resblock 
        self.pos_emb = SinusoidalPosEmb(dim=base_channels)
        time_dim = 4 * base_channels
        self.time_mlp = nn.Sequential(
            self.pos_emb, # B, T -> B, T, base_channels,
            nn.Linear(in_features=base_channels, out_features=4*base_channels), # B T base_channels * 4
            nn.GELU(),
            nn.Linear(in_features=4*base_channels, out_features=4*base_channels)
        ) # B T base_channels


        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.bottleneck = nn.ModuleList([])

        
        self.out_resblock = ResBlock(
                                in_channels=base_channels*2,
                                out_channels=base_channels,
                                groups=8, time_emb_dim=time_dim
        )
        self.out_proj = nn.Conv2d(
                            in_channels=base_channels,
                            out_channels=out_channels, 
                            kernel_size=1
        )


        # 32 64 128 256 512
        dims_list = [initial_channels, *map(lambda x: base_channels * x, dim_mults)]


        in_out = list(zip(dims_list[:-1], dims_list[1:]))
        # 32-64, 64-128, 128-256, 256-512 
        print(f"DIM LIST : {dims_list}")
        for i, (dim_in , dim_out) in enumerate(in_out):
            is_last = i >= (len(in_out) - 1)

            self.encoder.append(nn.ModuleList([
                ResBlock(in_channels=dim_in, out_channels=dim_in, groups=8, time_emb_dim=time_dim),
                ResBlock(in_channels=dim_in, out_channels=dim_in, groups=8, time_emb_dim=time_dim),
                Attention(dim=dim_in, heads=4, dim_head=32, norm_fn=RMSNorm),
                Downsample(dim=dim_in, dim_out=dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                
            ]))


        self.bottleneck.append(nn.ModuleList([
            ResBlock(dims_list[-1], dims_list[-1], groups=8, time_emb_dim=time_dim),
            Attention(dims_list[-1]),
            ResBlock(dims_list[-1], dims_list[-1], groups=8, time_emb_dim=time_dim)
        ]))

        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):

            is_last = i == (len(in_out) -1)

            self.decoder.append(nn.ModuleList([
                ResBlock(in_channels=dim_out + dim_in, out_channels=dim_out, groups=8, time_emb_dim=time_dim),
                ResBlock(in_channels=dim_out + dim_in, out_channels=dim_out, groups=8, time_emb_dim=time_dim),
                Attention(dim=dim_out),
                Upsample(dim=dim_out, dim_out=dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))



    def forward(self, x, t):

        x = self.in_proj(x)
        r = x.clone()

        t = self.time_mlp(t)

        skip_connections = []

        for b1, b2, attn, downsample in self.encoder:

            x = b1(x, t)
            skip_connections.append(x)
            x = b2(x, t)
            x = attn(x) + x
            skip_connections.append(x)
            x = downsample(x)

        res1_bottleneck, attn_bottleneck, res2_bottlenetck = self.bottleneck[0]

        x = res1_bottleneck(x, t)
        x = attn_bottleneck(x) + x
        x = res2_bottlenetck(x)


        for b1, b2, attn, upsample in self.decoder:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = b1(x, t)
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = b2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.out_resblock(x)

        return self.out_proj(x)

    
if __name__ == "__main__":
    # x = torch.ones((7, 3, 32, 32))

    # print(x.shape)

    # x = x.chunk(chunks=3, dim=1)
    # # tuple of tensors
    # print(len(x))
    # x = torch.stack(x)
    # print(x.shape)

    print(f"Dummy test UNET")

    x = torch.ones((7, 3, 32, 32))
    T = 1000

    t = torch.randint(low=1, high=T+1, size=(7,))

    print(f"input shape: {x.shape}")
    print(f"Time shape: {t.shape}")

    time_dim = 4 * 64

    pos_enc = SinusoidalPosEmb(dim=time_dim)

    time_emb = pos_enc(t)
    print(f"Sinusoal encoding shape : {time_emb.shape}")


    network = AttenionFiLMResUNet(
                    in_channels=3, 
                    out_channels=3,
                    initial_channels=64,
                    base_channels=64,
                    dim_mults=(1,2,4,8),
                    groups=8
    )

    print(f"Network init !")

    x = network(x, t)

    print(f"Network output shape : {x.shape}")