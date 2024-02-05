'''
    SA2T Transformer Model 
    Author: Stone
    Date: 24/05/2022
    Ref. https://github.com/FrancescoSaverioZuppichini/ViT
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange

import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb_size = args.tf_emb_size
        self.num_heads = args.head_num
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(args.token_emb, self.emb_size * 3 * self.num_heads)
        self.att_drop = nn.Dropout(args.dropout)
        self.projection = nn.Linear(self.emb_size * self.num_heads, self.emb_size)
        
    def forward(self, x : Tensor) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "n (h d qkv) -> (qkv) h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('hqd, hkd -> hqk', queries, keys) # batch, num_heads, query_len, key_len

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('hal, hlv -> hav ', att, values)
        out = rearrange(out, "h n d -> n (h d)")
        out = self.projection(out)

        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, args):
        super().__init__(
            nn.Linear(args.tf_emb_size, args.MLP_expansion * args.tf_emb_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.MLP_expansion * args.tf_emb_size, args.tf_emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, args):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(args.token_emb),
                MultiHeadAttention(args),
                nn.Dropout(args.dropout)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(args.tf_emb_size),
                FeedForwardBlock(args),
                nn.Dropout(args.dropout)
            )
            ))

class Transformer(nn.Sequential):
    def __init__(self, args, TF_depth):
        super().__init__()
        self.args = args
        self.tf_encoder_layers = [TransformerEncoderBlock(args).to(self.args.device) for _ in range(TF_depth)]

    def forward(self, x):
        for layer in self.tf_encoder_layers:
            x = layer(x)
        return x
