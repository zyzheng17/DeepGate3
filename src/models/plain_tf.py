'''
    Date: 24/05/2022
    Ref. https://github.com/FrancescoSaverioZuppichini/ViT
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn

from .mha import TransformerEncoderBlock

class Plain_Transformer(nn.Sequential):
    def __init__(self, args, TF_depth):
        super().__init__()
        self.args = args
        self.tf_encoder_layers = [TransformerEncoderBlock(args).to(self.args.device) for _ in range(TF_depth)]

    def forward(self, g, x):
        for layer in self.tf_encoder_layers:
            x = layer(x)
        return x
