from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn
import sys
from bert_model.transformer import TransformerBlock

class Path_Transformer(nn.Sequential):
    def __init__(self, args): 
        super().__init__()
        # Parameters
        self.args = args
        self.device = args.device
        
        # Model
        self.mask_token = nn.Parameter(torch.randn([args.token_emb,]))
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args.token_emb*2, args.head_num, args.token_emb*args.head_num, args.dropout) for _ in range(args.TF_depth)]
        )
        
    def forward(self, g, hs, hf): 
        hf = hf.detach()
        hs = hs.detach()
        bs = g.batch.max().item() + 1
        no_path = g.paths.shape[0]
        max_path_len = g.paths.shape[1]
        
        # mask po function embeddings 
        hf[g.hop_po.squeeze()] = self.mask_token
        
        for layer in range(self.args.TF_depth): 
            node_states = torch.cat([hs, hf], dim=1)
            next_hf = torch.zeros(hf.shape).to(self.device)
            for path_idx in range(no_path):
                path_nodes = g.paths[path_idx][:g.paths_len[path_idx]].long()
                path_node_states = node_states[path_nodes].unsqueeze(0)
                mask = torch.ones((len(path_nodes), len(path_nodes))).to(self.device)
                next_hf[path_nodes] += self.transformer_blocks[layer](path_node_states, mask)[0, :, self.args.token_emb:]
                
            hf = next_hf.detach().clone()
        
        return hf 