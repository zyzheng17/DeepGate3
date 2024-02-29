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
        self.mask_token = nn.Parameter(torch.randn([args.token_emb,]))
        # self.tf = TransformerEncoderBlock(args, args.token_emb*2).to(self.device)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args.token_emb*2, args.head_num, args.token_emb*args.head_num, args.dropout) for _ in range(args.TF_depth)]
        )
        
    def clean_record(self):
        self.record = {}
        
    def forward(self, g, hs, hf):
        hf = hf.detach().clone()
        hs = hs.detach().clone()
        next_hf = hf.detach().clone()
        bs = g.batch.max().item() + 1
        
        # mask po function embedding
        hf[g.hop_po.squeeze()] = self.mask_token
        
        for layer in range(self.args.TF_depth):
            node_states = torch.cat([hs, hf], dim=1)
            next_hf = torch.zeros(hf.shape).to(self.device)     # One node can be covered by some hops, sum up all states
            for hop_idx in range(len(g.hop_nodes)):
                hop_nodes = g.hop_nodes[hop_idx][g.hop_nodes_stats[hop_idx] == 1]
                hop_node_states = node_states[hop_nodes].unsqueeze(0)
                mask = torch.ones((len(hop_nodes), len(hop_nodes))).to(self.device)
                next_hf[hop_nodes] += self.transformer_blocks[layer](hop_node_states, mask)[0, :, self.args.token_emb:]
            hf = next_hf.detach().clone()
                    
        return hf 
                