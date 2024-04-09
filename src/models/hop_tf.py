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
# from utils.dag_utils import subgraph, subgraph_hop
# from bert_model.transformer import TransformerBlock

class Hop_Transformer(nn.Sequential):
    def __init__(self, args, hidden=128, attn_heads=4, dropout=0.1):
        super().__init__()
        # Parameters
        self.args = args
        self.hidden =hidden
        n_layers = self.args.TF_depth
        # Model
        # self.mask_token = nn.Parameter(torch.randn([args.token_emb,]))
        # self.tf = TransformerEncoderBlock(args, args.token_emb*2).to(self.device)
        # TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=args.token_emb*2, nhead=args.head_num, dropout=0.1, batch_first=True)
        # self.transformer_blocks = TransformerEncoderLayer

        TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=attn_heads, dropout=dropout, batch_first=True)
        self.function_transformer = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=n_layers)
        self.structure_transformer = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=n_layers)
        
    def clean_record(self):
        self.record = {}
        
    def forward(self, g, hf, hs):
        hf = hf.detach().clone()
        hs = hs.detach().clone()
        no_hops = g.winhop_nodes.shape[0]
        max_hop_size = torch.sum(g.winhop_nodes_stats, dim=1).max()
        g.winhop_nodes = g.winhop_nodes[:, :max_hop_size]
        g.winhop_nodes_stats = g.winhop_nodes_stats[:, :max_hop_size]
        
        # mask po function embedding
        # hf[g.winhop_po.squeeze()] = self.mask_token
        
        # Hop TF
        for level in range(g.forward_level.max()):
            hop_mask = g.forward_level[g.winhop_po] == level
            level_hop_index = torch.nonzero(hop_mask.squeeze()).view(-1)
            no_level_hops = len(level_hop_index)
            if no_level_hops == 0:
                continue
            
            # node_states = torch.cat([hs, hf], dim=1)
            node_states = hf + hs
            node_idx = g.winhop_nodes[level_hop_index]
            hop_node_emb = node_states[node_idx]
            masks = g.winhop_nodes_stats[level_hop_index]
            masks = torch.where(masks==1,False,True)
            hf_states = self.function_transformer(hop_node_emb, src_key_padding_mask=masks)
            
            # Stone: Modify the structure transformer
            hs_states = self.structure_transformer(hs[node_idx], src_key_padding_mask=masks)
            # hs_states = self.structure_transformer(hop_node_emb, src_key_padding_mask=masks)
            
            hop_node_idx = node_idx[g.winhop_nodes_stats[level_hop_index]==1]
            hf[hop_node_idx] = hf_states[g.winhop_nodes_stats[level_hop_index]==1]
            hs[hop_node_idx] = hs_states[g.winhop_nodes_stats[level_hop_index]==1]
            
        return hf,hs
                