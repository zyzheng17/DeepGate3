import torch 
import deepgate as dg
import torch.nn as nn 
from .mlp import MLP

class PoolNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pool = MLP(
            dim_in=args.dim_hidden * args.no_pi + args.dim_hidden * args.no_po,
            dim_hidden=args.mlp_hidden, 
            dim_pred=args.dim_hidden,
            num_layer=args.mlp_layer,
            norm_layer=args.norm_layer,
            act_layer=args.act_layer,
        )
        
    def forward(self, x):
        return self.pool(x)
        