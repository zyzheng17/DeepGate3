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
import copy


from .mha import TransformerEncoderBlock
from utils.dag_utils import subgraph, subgraph_hop
from bert_model.transformer import TransformerBlock


class Hop_Transformer(nn.Sequential):
    def __init__(self, args):
        super().__init__()
        # Parameters
        self.args = args
        self.device = args.device
        
        # Model
        self.tf_hf = [TransformerEncoderBlock(args, args.token_emb*2).to(self.device) for _ in range(self.args.TF_depth)]
        self.tf_hs = [TransformerEncoderBlock(args, args.token_emb).to(self.device) for _ in range(self.args.TF_depth)]
        
    def clean_record(self):
        self.record = {}
        
    def forward(self, hs, hf, subgraph):
        next_hs = hs.detach().clone()
        hs = hs.detach().clone()
        next_hf = hf.detach().clone()
        hf = hf.detach().clone()
        
        for layer in range(self.args.TF_depth):
            for idx in range(len(hs)):
                if idx not in subgraph.keys():
                    continue
                hop_nodes = subgraph[idx]['nodes']
                hop_nodes = hop_nodes.long().to(self.device)
                next_hs[idx] = self.tf_hs[layer](hs[hop_nodes])[-1]
                next_hf[idx] = self.tf_hf[layer](torch.cat([hs[hop_nodes], hf[hop_nodes]], dim=-1))[-1,hs.shape[1]:]
            hs = next_hs.detach().clone()
            hf = next_hf.detach().clone()
        
        return hs, hf 
                