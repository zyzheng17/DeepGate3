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
        self.hidden = hidden
        self.record = {}
        self.mask_token = nn.Parameter(torch.randn([hidden,]))
        #TODO: the max_length should be the max number of gate in a circuit
        self.max_length = 512
        TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=attn_heads, dropout=dropout)
        self.transformer_blocks = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=n_layers)

        # self.transformer_blocks = nn.ModuleList(
        #     [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.cls_head = nn.Sequential(nn.Linear(hidden*2, hidden*4),
                        nn.ReLU(),
                        nn.LayerNorm(hidden*4),
                        nn.Linear(hidden*4, 16))

    def clean_record(self):
        self.record = {}

    # def forward(self, g, subgraph):
    def forward(self, g, hs, hf):
        hf = hf.detach()
        hs = hs.detach()
        bs = g.batch.max().item() + 1
        #mask po function embedding
        # hf[g.all_hop_po.squeeze()] = self.mask_token
        
        mask_hop_states = torch.zeros([bs,self.max_length,self.hidden]).to(hf.device)
        mask = torch.ones([bs,self.max_length]).to(hf.device)
        for i in range(bs):
            # batch_idx = torch.argwhere(g.batch==i).squeeze(-1)
            # # add hs as positional embedding
            # mask_hop_states[i] = torch.cat([hf[batch_idx] + hs[batch_idx], \
            #                                 torch.zeros([self.max_length - hf[batch_idx].shape[0],hf[batch_idx].shape[1]]).to(hf.device)],dim=0)
            mask_hop_states[i] = torch.cat([hf[g.batch==i] + hs[g.batch==i], \
                                            torch.zeros([self.max_length - hf[g.batch==i].shape[0],hf[g.batch==i].shape[1]]).to(hf.device)],dim=0)
            mask[i][:hf[g.batch==i].shape[0]] = 0

        # mask = (mask_hop_states[:,:,0] != 0).unsqueeze(1).repeat(1, mask_hop_states.shape[1], 1).unsqueeze(1) 
        # for transformer in self.transformer_blocks:
        #     mask_hop_states = transformer.forward(mask_hop_states, mask)
        mask_hop_states = mask_hop_states.permute(1,0,2)
        mask_hop_states = self.transformer_blocks(mask_hop_states, src_key_padding_mask=mask)
        mask_hop_states = mask_hop_states.permute(1,0,2)
        for i in range(bs):
            # batch_idx = torch.argwhere(g.batch==i).squeeze(-1)
            batch_idx = g.forward_index[g.batch==i]
            hf[batch_idx] = mask_hop_states[i,:batch_idx.shape[0]]
        
        return hf
