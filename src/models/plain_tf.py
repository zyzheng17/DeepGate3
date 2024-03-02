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
        self.num_head = attn_heads
        self.mask_token = nn.Parameter(torch.randn([hidden,]))
        #TODO: the max_length should be the max number of gate in a circuit
        self.max_length = 512
        TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=attn_heads, dropout=dropout, batch_first=True)
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
        corr_m = g.fanin_fanout_cone.reshape(bs, self.max_length, self.max_length)
        
        corr_m = torch.where(corr_m == 0, True, False) # Flase = compute attention, True = mask # inverse to fit nn.transformer
        #multi-head attention: len, bs, emb -> len, bs*numhead, head_emb by tensor.reshape
        #corr-mask: bs,len,len
        bs,l1,l2 = corr_m.shape
        corr_m = corr_m.unsqueeze(1).repeat(1,self.num_head,1,1).reshape(bs*self.num_head,l1,l2)
        
        #mask po function embedding
        # hf[g.all_hop_po.squeeze()] = self.mask_token
        
        mask_hop_states = torch.zeros([bs,self.max_length,self.hidden]).to(hf.device)
        padding_mask = torch.ones([bs,self.max_length]).to(hf.device)
        for i in range(bs):
            mask_hop_states[i] = torch.cat([hf[g.batch==i] + hs[g.batch==i], \
                                            torch.zeros([self.max_length - hf[g.batch==i].shape[0],hf[g.batch==i].shape[1]]).to(hf.device)],dim=0)
            padding_mask[i][:hf[g.batch==i].shape[0]] = 0

        padding_mask = torch.where(padding_mask==1, True, False)# Flase = compute attention, True = mask # inverse to fit nn.transformer
        padding_mask1 = padding_mask.view(bs,1,1,self.max_length).expand(-1,self.num_head,-1,-1).reshape(bs*self.num_head,1,self.max_length)
        # mask1_b = corr_m or padding_mask1
        mask1 = torch.where(corr_m == 1, -torch.inf,0) + torch.where(padding_mask1 == 1, -torch.inf, 0)
        mask1 = torch.logical_or(corr_m,padding_mask1)
        # key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
        #     expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        for i in range(64):
            for j in range(512):
                if corr_m[i][j].sum() == 512:
                    print(f'{i} {j} corr_m all True')
                if mask1[i][j].sum() == 512:
                    print(f'{i} {j} mask1 all True')
                
        mask_hop_states = self.transformer_blocks(mask_hop_states, src_key_padding_mask=padding_mask, mask = corr_m)

        for i in range(bs):
            batch_idx = g.forward_index[g.batch==i]
            hf[batch_idx] = mask_hop_states[i,:batch_idx.shape[0]]
        
        return hf
