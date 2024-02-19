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
# class Plain_Transformer(nn.Sequential):
#     def __init__(self, args, TF_depth):
#         super().__init__()
#         self.args = args
#         self.tf_encoder_layers = [TransformerEncoderBlock(args).to(self.args.device) for _ in range(TF_depth)]

#     def forward(self, g, x):
#         for layer in self.tf_encoder_layers:
#             x = layer(x)
#         return x
    
class Plain_Transformer(nn.Sequential):
    def __init__(self, args, hidden=128, n_layers=12, attn_heads=4, dropout=0.1):
        super().__init__()
        self.args = args
        # self.tf_encoder_layers = [TransformerEncoderBlock(args).to(self.args.device) for _ in range(TF_depth)]
        self.record = {}
        self.mask_token = nn.Parameter(torch.zeros([1,]))
        #TODO: the max_length should be the max number of gate in a circuit
        self.max_length = 512
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.cls_head = nn.Sequential(nn.Linear(hidden*2, hidden*4),
                        nn.ReLU(),
                        nn.LayerNorm(hidden*4),
                        nn.Linear(hidden*4, 16))

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
            mask_node_states[hop_nodes_idx] = self.mask_token
            mask_hop_states = mask_node_states[torch.argwhere(g.batch==batch_idx)].squeeze(1)
            mask_hop_states = torch.cat([mask_hop_states, \
                                          torch.zeros([self.max_length - mask_hop_states.shape[0],mask_hop_states.shape[1]]).to(mask_hop_states.device)],dim=0)
            mask_hop_states_list.append(mask_hop_states.unsqueeze(0))
            
            current_pi_idx = subgraph[k]['pis']-len(torch.argwhere(g.batch<batch_idx))
            current_po_idx = subgraph[k]['pos']-len(torch.argwhere(g.batch<batch_idx))
            pis.append(current_pi_idx)
            pos.append(current_po_idx)

        mask_hop_states_list = torch.cat(mask_hop_states_list,dim=0) # bs x seq_len x emb_len

        mask = (mask_hop_states_list[:,:,0] > 0).unsqueeze(1).repeat(1, mask_hop_states_list.size(1), 1).unsqueeze(1) 
        for transformer in self.transformer_blocks:
            mask_hop_states_list = transformer.forward(mask_hop_states_list, mask)
        
        for i,k in enumerate(list(subgraph.keys())):
            hf_dict[k] = {}
            hf_dict[k]['pi_emb'] = mask_hop_states_list[i][pis[i]]
            hf_dict[k]['po_emb'] = mask_hop_states_list[i][pos[i]]
            
        return hf_dict