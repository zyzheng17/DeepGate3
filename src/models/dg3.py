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
        self.hs_mask_token = nn.Parameter(torch.zeros([self.args.token_emb,]))
        self.hf_mask_token = nn.Parameter(torch.zeros([self.args.token_emb,]))
        self.mask_pred_hs = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=self.args.token_emb, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer=self.args.act_layer
        )
        self.mask_pred_hf = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=self.args.token_emb, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer=self.args.act_layer
        )
        self.hs_pool = tf_Pooling(args)
        self.hf_pool = tf_Pooling(args)
        self.tt_pred = [nn.Sequential(nn.Linear(self.args.token_emb, 1), nn.Sigmoid()) for _ in range(self.max_tt_len)]
        self.prob_pred = nn.Sequential(nn.Linear(self.args.token_emb, self.args.token_emb), nn.ReLU(), nn.Linear(self.args.token_emb, 1), nn.ReLU())
        
        self.cls_head = nn.Sequential(nn.Linear(self.hidden, self.hidden*4),
                        nn.ReLU(),
                        nn.LayerNorm(self.hidden*4),
                        nn.Linear(self.hidden*4, self.max_tt_len))



    def forward(self, g, subgraph):
        
        # Tokenizer
        hs, hf = self.tokenizer(g)
        
        # Transformer 
        # tf_hs, tf_hf = hs, hf
        bs = g.batch[-1]+1
        # tf_hs, tf_hf = self.transformer(g, hs, hf, subgraph)
        hf_dict = self.transformer(g, hs, hf, subgraph)
        
        # Pooling 
        hop_hs = None
        # hop_hs = torch.zeros(len(g.gate), self.args.token_emb).to(self.args.device)
        # hop_hf = torch.zeros(len(g.gate), self.args.token_emb).to(self.args.device)
        hop_hf = {}
        logits = {}
        for idx in subgraph.keys():
            # hop_hs[idx] = self.hs_pool(torch.cat([tf_hs[subgraph[idx]['pos'].long()], tf_hs[subgraph[idx]['pis'].long()]], dim=0))
            # hop_hf[idx] = self.hf_pool(torch.cat([tf_hf[subgraph[idx]['pos'].long()], tf_hf[subgraph[idx]['pis'].long()]], dim=0))


            #transformer pooling
            # hop_hf[idx] = self.hf_pool(torch.cat([hf_dict[idx]['pi_emb'],hf_dict[idx]['po_emb']], dim=0))

            #average pooling
            hop_hf[idx] = torch.mean(torch.cat([hf_dict[idx]['pi_emb'],hf_dict[idx]['po_emb']], dim=0))
            
            logits[idx] = self.cls_head(hop_hf[idx])
        
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