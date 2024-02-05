import torch 
import deepgate as dg
import torch.nn as nn 
from .pool import PoolNet
from .mlp import MLP
from .dg2 import DeepGate2
from .transformer import Transformer

class DeepGate3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Tokenizer
        self.tokenizer = DeepGate2()
        self.tokenizer.load_pretrained(args.pretrained_model_path)
        
        # Hop Transformer 
        self.transformer = Transformer(args, args.TF_depth)
        
    def forward(self, g):
        hs, hf = self.tokenizer(g)
        pe = hs 
        node_state = torch.cat([hf, pe], dim=-1)
        node_state = self.transformer(node_state)
        
        return node_state