import torch
import torch.nn.functional as F
from torch import nn
import copy


from .mha import MultiHeadAttention
from utils.dag_utils import subgraph, subgraph_hop
from bert_model.transformer import TransformerBlock

class tf_Pooling(nn.Module):
    def __init__(self, args): 
        super().__init__()
        self.args = args
        self.dim = args.token_emb 
        self.tf = MultiHeadAttention(args, self.dim)
        self.cls_token = nn.Parameter(torch.zeros([1, self.dim]))
        
    def forward(self, x):
        states = torch.cat([self.cls_token, x], dim=0)
        states = self.tf(states)
        return states[0]
        