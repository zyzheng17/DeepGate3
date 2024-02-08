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

class Hop_Transformer(nn.Sequential):
    def __init__(self, args, TF_depth):
        super().__init__()
        self.args = args
        self.tf_encoder_layers = [TransformerEncoderBlock(args).to(self.args.device) for _ in range(TF_depth)]
        self.record = {}
        
    def clean_record(self):
        self.record = {}
        
    def forward(self, g, x):
        last_node_states = x.detach().clone()
        node_states = x.detach().clone()
        if g not in self.record and self.args.hop_record:
            self.record[g] = {}
        
        for layer in self.tf_encoder_layers:
            for idx in range(len(g.gate)):
                if self.args.hop_record and idx in self.record[g]:
                    hop_nodes = self.record[g][idx]
                if self.args.hop_record and idx not in self.record[g]:
                    hop_nodes = subgraph_hop([idx], g.edge_index, hops=3, dim=1)
                    self.record[g][idx] = hop_nodes
                if not self.args.hop_record:
                    hop_nodes = subgraph_hop([idx], g.edge_index, hops=3, dim=1)
                    
                hop_nodes = hop_nodes.to(self.args.device)
                # TODO: Consider update node_states[idx] instead of node_states[hop_nodes]
                node_states[hop_nodes] = layer(last_node_states[hop_nodes])
            last_node_states = node_states.detach().clone()

        return node_states
