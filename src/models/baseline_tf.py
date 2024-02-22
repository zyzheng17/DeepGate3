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

# from .mha import TransformerEncoderBlock
import sys
# sys.path.append()
from bert_model.transformer import TransformerBlock

    
class Baseline_Transformer(nn.Sequential):
    def __init__(self, args, hidden=128, n_layers=12, attn_heads=4, dropout=0.1):
        super().__init__()
        self.args = args
        # self.tf_encoder_layers = [TransformerEncoderBlock(args).to(self.args.device) for _ in range(TF_depth)]
        self.record = {}
        self.mask_token = nn.Parameter(torch.zeros([1,]))
        #TODO: the max_length should be the max number of gate in a circuit
        self.max_length = 512

    def clean_record(self):
        self.record = {}

    # def forward(self, g, subgraph):
    def forward(self, g, hs, hf, subgraph):
        initial_node_states = hf.detach().clone()

        # batch_idx = torch.randint(0,len(g.gate),[128,])
        # Mask Graph Modeling
        # random mask a hop and predict it
        logits = []
        tts = []
        hf_dict = {}
        mask_hop_states_list=[]
        pis = []
        pos = []
        hf = hf.detach()
        for k in list(subgraph.keys()):
            hop_nodes_idx = subgraph[k]['nodes']
            batch_idx = g.batch[k]
            mask_node_states = hf.clone()
            # mask_node_states[hop_nodes_idx] = self.mask_token
            mask_hop_states = mask_node_states[torch.argwhere(g.batch==batch_idx)].squeeze(1)
            mask_hop_states = torch.cat([mask_hop_states, \
                                          torch.zeros([self.max_length - mask_hop_states.shape[0],mask_hop_states.shape[1]]).to(mask_hop_states.device)],dim=0)
            mask_hop_states_list.append(mask_hop_states.unsqueeze(0))
            
            current_pi_idx = subgraph[k]['pis']-len(torch.argwhere(g.batch<batch_idx))
            current_po_idx = subgraph[k]['pos']-len(torch.argwhere(g.batch<batch_idx))
            pis.append(current_pi_idx)
            pos.append(current_po_idx)

        mask_hop_states_list = torch.cat(mask_hop_states_list,dim=0) # bs x seq_len x emb_len
        
        for i,k in enumerate(list(subgraph.keys())):
            hf_dict[k] = {}
            hf_dict[k]['pi_emb'] = mask_hop_states_list[i][pis[i]]
            hf_dict[k]['po_emb'] = mask_hop_states_list[i][pos[i]]
            
        return hf_dict