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
    def __init__(self, args, hidden=128, n_layers=9, attn_heads=4, dropout=0.1):
        super().__init__()
        self.args = args
        self.hidden = hidden
        self.record = {}
        self.num_head = attn_heads
        self.max_length = 512
        TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=attn_heads, dropout=dropout, batch_first=True)
        self.function_transformer = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=n_layers)
        self.structure_transformer = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=n_layers)



    # def forward(self, g, subgraph):
    def forward(self, g, hs, hf):
        hf = hf.detach()
        hs = hs.detach()
        bs = g.batch.max().item() + 1
        corr_m = g.fanin_fanout_cone.reshape(bs, self.max_length, self.max_length)
        
        corr_m = torch.where(corr_m == 0, True, False) # Flase = compute attention, True = mask # inverse to fit nn.transformer
        #multi-head attention: len, bs, emb -> len, bs*numhead, head_emb by tensor.reshape
        #corr-mask: bs,len,len
        bs,l1,l2 = corr_m.shape
        corr_m = corr_m.unsqueeze(1).repeat(1,self.num_head,1,1).reshape(bs*self.num_head,l1,l2)
        
        mask_hop_states = torch.zeros([bs,self.max_length,self.hidden]).to(hf.device)
        padding_mask = torch.ones([bs,self.max_length]).to(hf.device)
        for i in range(bs):
            mask_hop_states[i] = torch.cat([hf[g.batch==i] + hs[g.batch==i], \
                                            torch.zeros([self.max_length - hf[g.batch==i].shape[0],hf[g.batch==i].shape[1]]).to(hf.device)],dim=0)
            padding_mask[i][:hf[g.batch==i].shape[0]] = 0

        padding_mask = torch.where(padding_mask==1, True, False)# Flase = compute attention, True = mask # inverse to fit nn.transformer
                
        hf_tf = self.function_transformer(mask_hop_states, src_key_padding_mask=padding_mask, mask = corr_m)
        hs_tf = self.structure_transformer(mask_hop_states, src_key_padding_mask=padding_mask, mask = corr_m)


        for i in range(bs):
            batch_idx = g.forward_index[g.batch==i]
            hf[batch_idx] = hf_tf[i,:batch_idx.shape[0]]
            hs[batch_idx] = hs_tf[i,:batch_idx.shape[0]]
        
        return hf, hs
