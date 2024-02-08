import torch 
import deepgate as dg
import torch.nn as nn 
from .pool import PoolNet
from .mlp import MLP
from .dg2 import DeepGate2

from .plain_tf import Plain_Transformer
from .hop_tf import Hop_Transformer

_transformer_factory = {
    'plain': Plain_Transformer,
    'hop': Hop_Transformer, 
}

class DeepGate3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Tokenizer
        self.tokenizer = DeepGate2()
        self.tokenizer.load_pretrained(args.pretrained_model_path)
        
        # Transformer 
        self.transformer = _transformer_factory[args.tf_arch](args, args.TF_depth)
        
    def forward(self, g):
        hs, hf = self.tokenizer(g)
        pe = hs 
        node_state = torch.cat([hf, pe], dim=-1)
        node_state = self.transformer(g, node_state)
        
        return node_state