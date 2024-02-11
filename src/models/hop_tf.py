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
        self.args = args
        # self.tf_encoder_layers = [TransformerEncoderBlock(args).to(self.args.device) for _ in range(TF_depth)]
        self.record = {}
        self.mask_token = nn.Parameter(torch.zeros([1,]))
        #TODO: the max_length should be the max number of gate in a circuit
        self.max_length = 2048
        self.transformer = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.cls_head = nn.Sequential(nn.Linear(hidden*2, hidden*4),
                        nn.ReLU(),
                        nn.LayerNorm(hidden*4),
                        nn.Linear(hidden*4, 16))

    def clean_record(self):
        self.record = {}
        
    def forward(self, g, x):
        last_node_states = x.detach().clone()
        initial_node_states = x.detach().clone()
        
        if g not in self.record and self.args.hop_record:
            self.record[g] = {}
        
        # for layer in self.tf_encoder_layers:

        #     for idx in range(len(g.gate)):
        #         if self.args.hop_record and idx in self.record[g]:
        #             hop_nodes = self.record[g][idx]
        #         if self.args.hop_record and idx not in self.record[g]:
        #             hop_nodes = subgraph_hop([idx], g.edge_index, hops=3, dim=1)
        #             self.record[g][idx] = hop_nodes
        #         if not self.args.hop_record:
        #             hop_nodes = subgraph_hop([idx], g.edge_index, hops=3, dim=1)
                    
        #         hop_nodes = hop_nodes.to(self.args.device)
        #         # TODO: Consider update node_states[idx] instead of node_states[hop_nodes]
        #         node_states[hop_nodes] = layer(last_node_states[hop_nodes])
        #     last_node_states = node_states.detach().clone()

        
        batch_idx = torch.randint(0,len(g.gate),[128,])
        # Mask Graph Modeling
        # random mask a hop and predict it
        logits = []
        tts = []
        for idx in batch_idx:
            if self.args.hop_record and idx in self.record[g]:
                #TODO: here hop_nodes should be the index of the k-hop
                #TODO: need to get the truth table of the hop
                hop_nodes = self.record[g][idx]
            if self.args.hop_record and idx not in self.record[g]:
                hop_nodes = subgraph_hop([idx], g.edge_index, hops=3, dim=1)
                self.record[g][idx] = hop_nodes
            if not self.args.hop_record:
                hop_nodes = subgraph_hop([idx], g.edge_index, hops=3, dim=1)
            
            mask_node_states = initial_node_states.clone() # length x 128
            mask_node_states[hop_nodes] = self.mask_token
            # pad to max length with zero
            mask_node_states = torch.cat([mask_node_states,torch.zeros([self.max_length - mask_node_states.shape[0],mask_node_states.shape[1]]).to(self.device)],dim=0)
            # attention mask for pedding zero, not mask token
            mask = (mask_node_states > 0).unsqueeze(1).repeat(1, mask_node_states.size(1), 1).unsqueeze(1) 
            for transformer in self.transformer_blocks:
                mask_node_states = transformer.forward(mask_node_states, mask)
            
            #truth table prediction
            hop_node_state = mask_node_states[hop_nodes]

            #TODO: need to get PI PO idx
            PI = torch.mean(hop_node_state[hop_PI_idx], dim=0)
            PO = torch.mean(hop_node_state[hop_PO_idx], dim=0)

            hop_pool = torch.cat([PI,PO])

            logit = self.cls_head(hop_pool)
            logits.append(logit)
            tts.append(tt)

        logits = torch.stack(logits)
        tts = torch.stack(tts)
        return logits, tts





            














            




        return node_states
