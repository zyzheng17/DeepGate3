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
    def __init__(self, args, hidden=128, n_layers=12, attn_heads=4, dropout=0.1):
        super().__init__()
        # Parameters
        self.args = args
        self.device = args.device
        self.hidden =hidden
        # Model
        self.mask_token = nn.Parameter(torch.randn([args.token_emb,]))
        # self.tf = TransformerEncoderBlock(args, args.token_emb*2).to(self.device)
        # TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=args.token_emb*2, nhead=args.head_num, dropout=0.1, batch_first=True)
        # self.transformer_blocks = TransformerEncoderLayer

        TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=attn_heads, dropout=dropout, batch_first=True)
        self.function_transformer = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=n_layers)
        self.structure_transformer = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=n_layers)


        
    def clean_record(self):
        self.record = {}
        
    def forward(self, g, hs, hf):
        hf = hf.detach().clone()
        hs = hs.detach().clone()
        no_hops = g.hop_nodes.shape[0]
        max_hop_size = g.hop_nodes.shape[1]
        
        # mask po function embedding
        # hf[g.hop_po.squeeze()] = self.mask_token
        
        # Hop TF
        for level in range(g.forward_level.max()):
            hop_mask = g.forward_level[g.hop_po] == level
            level_hop_index = torch.nonzero(hop_mask.squeeze()).view(-1)
            no_level_hops = len(level_hop_index)
            if no_level_hops == 0:
                continue
            
            ########################################################
            # For-loop Implementation
            ########################################################
            node_states = torch.cat([hs, hf], dim=1)
            for hop_idx in range(no_level_hops):
                no_nodes_in_hop = g.hop_nodes_stats[level_hop_index[hop_idx]].sum()
                hop_nodes = g.hop_nodes[level_hop_index[hop_idx]][:no_nodes_in_hop]
                hop_node_states = node_states[hop_nodes]
                hop_node_states = hop_node_states.unsqueeze(0)
                hf_states = self.function_transformer(hop_node_states, src_key_padding_mask=None, src_mask=None)
                hf[hop_nodes] += hf_states[:, :no_nodes_in_hop, self.args.token_emb:].squeeze(0)
            
            
            ########################################################
            # TODO: Batch Implementation
            # Find Nan in hop_node_states
            ########################################################
            # # Prepare mask
            # padding_mask = torch.zeros(no_level_hops, max_hop_size).to(self.device)
            # node_mask = torch.zeros(0, max_hop_size, max_hop_size).to(self.device)
            # hop_nodes = torch.zeros(0, max_hop_size).to(self.device)
            # for hop_idx in range(no_level_hops):
            #     no_nodes_in_hop = g.hop_nodes_stats[level_hop_index[hop_idx]].sum()
            #     padding_mask[hop_idx, no_nodes_in_hop:] = 1
            #     hop_nodes = torch.cat([hop_nodes, g.hop_nodes[level_hop_index[hop_idx]].unsqueeze(0)], dim=0)
            #     mask = torch.zeros(no_nodes_in_hop, no_nodes_in_hop).to(self.device)
            #     mask = torch.nn.functional.pad(mask, (0, max_hop_size-no_nodes_in_hop, 0, max_hop_size-no_nodes_in_hop), value=1)
            #     node_mask = torch.cat([node_mask, mask.unsqueeze(0)], dim=0)
            # node_mask = node_mask.repeat(self.args.head_num, 1, 1)
            # hop_nodes = hop_nodes.long()
            
            # # Transformer 
            # node_states = torch.cat([hs, hf], dim=1)
            # hop_node_states = node_states[hop_nodes]
            # padding_mask = padding_mask.bool()
            # node_mask = node_mask.bool()
            # hop_node_states = self.transformer_blocks(hop_node_states, src_key_padding_mask=padding_mask, src_mask=node_mask)
            # for hop_idx in range(no_level_hops):
            #     hop_valid_nodes = g.hop_nodes[level_hop_index[hop_idx]][g.hop_nodes_stats[level_hop_index[hop_idx]] == 1]
            #     no_nodes_in_hop = len(hop_valid_nodes)
            #     hf[hop_valid_nodes] += hop_node_states[hop_idx, :no_nodes_in_hop, self.args.token_emb:]            
            
        return hf 
                