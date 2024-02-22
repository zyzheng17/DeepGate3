import torch 
import deepgate as dg
import torch.nn as nn 
# from .pool import PoolNet
import sys
# sys.path.append('/uac/gds/zyzheng23/projects/DeepGate3-Transformer/src/models/dg3.py')
# sys.path.append('/uac/gds/zyzheng23/projects/DeepGate3-Transformer/src/models')
# sys.path.append('/uac/gds/zyzheng23/projects/DeepGate3-Transformer/src')
from .mlp import MLP
from .dg2 import DeepGate2

from .plain_tf import Plain_Transformer
from .hop_tf import Hop_Transformer
from .baseline_tf import Baseline_Transformer
from .mlp import MLP
from .tf_pool import tf_Pooling

_transformer_factory = {
    'baseline': Baseline_Transformer,
    'plain': Plain_Transformer,
    'hop': Hop_Transformer, 
}

import torch.nn as nn

class DeepGate3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_tt_len = 64
        self.hidden = 128
        
        # Tokenizer
        self.tokenizer = DeepGate2()
        self.tokenizer.load_pretrained(args.pretrained_model_path)
        
        # Transformer 
        self.transformer = _transformer_factory[args.tf_arch](args)
        
        # Prediction 
        self.mask_pred_hs = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=self.args.token_emb, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer=self.args.act_layer
        )
        self.mask_pred_hf = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=self.args.token_emb, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer=self.args.act_layer
        )

        #pooling layer
        # pool_layer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=4)
        # self.tf_encoder = nn.TransformerEncoder(pool_layer, num_layers=3)

        self.hs_pool = tf_Pooling(args)
        self.hf_pool = tf_Pooling(args)
        self.tt_pred = [nn.Sequential(nn.Linear(self.args.token_emb, 1), nn.Sigmoid()) for _ in range(self.max_tt_len)]
        self.prob_pred = nn.Sequential(nn.Linear(self.args.token_emb, self.args.token_emb), nn.ReLU(), nn.Linear(self.args.token_emb, 1), nn.ReLU())
        
        self.cls_head = nn.Sequential(nn.Linear(self.hidden, self.hidden*4),
                        nn.ReLU(),
                        nn.LayerNorm(self.hidden*4),
                        nn.Linear(self.hidden*4, self.max_tt_len))



    def forward(self, g):
        bs = g.batch_size
        # subgraph = {}
        # subgraph['pi'] = g.all_hop_pi
        # subgraph['po'] = g.all_hop_po
        # subgraph['pi_stats'] = g.all_hop_pi_stats
        # subgraph['tt'] = g.all_tt
        # Tokenizer
        hs, hf = self.tokenizer(g)
        
        # Refine-Transformer 
        hf = self.transformer(g, hs, hf)

        #gate-level pretrain task : predict global probability
        #TODO

        #graph-level pretrain task : predict truth table
        hop_hf = []
        for i in range(g.all_hop_po.shape[0]):
            pi_idx = g.all_hop_pi[i][torch.argwhere(g.all_hop_pi_stats[i]!=-1)].squeeze(-1)
            hop_hf.append( self.hf_pool(torch.cat([hf[pi_idx],hf[g.all_hop_po[i]]], dim=0)) )

        hop_hf = torch.stack(hop_hf)
        logits = self.cls_head(hop_hf)
        

        
        return logits
    
    def pred_tt(self, graph_emb, no_pi):
        tt = []
        for pi in range(self.max_tt_len):
            tt.append(self.tt_pred[pi](graph_emb))
        tt = torch.tensor(tt).squeeze()
        return tt
    
    def pred_prob(self, hf):
        prob = self.prob_pred(hf)
        return prob