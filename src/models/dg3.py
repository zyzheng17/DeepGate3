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
    'baseline': None,
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
        self.tf_arch = args.tf_arch
        # Tokenizer
        self.tokenizer = DeepGate2()
        self.tokenizer.load_pretrained(args.pretrained_model_path)

        #special token
        self.cls_token = nn.Parameter(torch.randn([self.hidden,]))
        self.dc_token = nn.Parameter(torch.randn([self.hidden,]))
        self.zero_token = nn.Parameter(torch.randn([self.hidden,]))
        self.one_token = nn.Parameter(torch.randn([self.hidden,]))
        self.pad_token = torch.zeros([self.hidden,]) # dont learn
        self.pool_max_length = 10
        self.PositionalEmbedding = nn.Embedding(10,self.hidden)
        
        # Transformer 
        if args.tf_arch != 'baseline':
            self.transformer = _transformer_factory[args.tf_arch](args)
        
        # Prediction 
        self.readout_prob = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )

        #pooling layer
        pool_layer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=4)
        self.tf_encoder = nn.TransformerEncoder(pool_layer, num_layers=3)

        # self.hs_pool = tf_Pooling(args)
        # self.hf_pool = tf_Pooling(args)
        # self.tt_pred = [nn.Sequential(nn.Linear(self.args.token_emb, 1), nn.Sigmoid()) for _ in range(self.max_tt_len)]
        # self.prob_pred = nn.Sequential(nn.Linear(self.args.token_emb, self.args.token_emb), nn.ReLU(), nn.Linear(self.args.token_emb, 1), nn.ReLU())
        
        self.cls_head = nn.Sequential(nn.Linear(self.hidden, self.hidden*4),
                        nn.ReLU(),
                        nn.LayerNorm(self.hidden*4),
                        nn.Linear(self.hidden*4, self.max_tt_len))



    def forward(self, g):
        bs = g.batch.max().item() + 1
        hs, hf = self.tokenizer(g)
        hf = hf.detach()
        hs = hs.detach()
        # Refine-Transformer 
        if self.tf_arch != 'baseline':
            #non-residual
            # hf = self.transformer(g, hs, hf)

            #with-residual
            hf = hf + self.transformer(g, hs, hf)

        #gate-level pretrain task : predict global probability
        prob = self.readout_prob(hf)

        #graph-level pretrain task : predict truth table
        hop_hf = []
        masks = []

        for i in range(g.hop_po.shape[0]):
            pi_idx = g.hop_pi[i][g.hop_pi_stats[i]!=-1].squeeze(-1)
            pi_hop_stats = g.hop_pi_stats[i]
            pi_emb = hf[pi_idx]
            pi_emb = []
            for j in range(8):
                if pi_hop_stats[j] == -1:
                    continue
                elif pi_hop_stats[j] == 0:
                    pi_emb.append(self.zero_token)
                elif pi_hop_stats[j] == 1:
                    pi_emb.append(self.one_token)
                elif pi_hop_stats[j] == 2:
                    pi_emb.append(hf[g.hop_pi[i][j]])
            # add dont care token
            while len(pi_emb) < 6:
                pi_emb.insert(0,self.dc_token)
            # pad seq to fixed length
            mask = [1 for _ in range(len(pi_emb))]
            while len(pi_emb) < 8:
                pi_emb.append(self.pad_token.to(hf.device))
                mask.append(0)

            pi_emb = torch.stack(pi_emb) # 8 128
            po_emb = hf[g.hop_po[i]] # 1 128
            hop_hf.append(torch.cat([self.cls_token.unsqueeze(0),pi_emb,po_emb], dim=0)) 
            mask.insert(0,1)
            mask.append(1)
            masks.append(torch.tensor(mask))

        hop_hf = torch.stack(hop_hf) #bs seq_len hidden
        pos = torch.arange(hop_hf.shape[1]).unsqueeze(0).repeat(hop_hf.shape[0],1).to(hf.device)
        hop_hf = hop_hf + self.PositionalEmbedding(pos)

        hop_hf = hop_hf.permute(1,0,2)#seq_len bs hidden
        masks = 1 - torch.stack(masks).to(hf.device).float() #bs seq_len 
        
        hop_hf = self.tf_encoder(hop_hf,src_key_padding_mask = masks.float())
        hop_hf = hop_hf.permute(1,0,2)[:,0]
        hop_tt = self.cls_head(hop_hf)
        
        return hs, hf, prob, hop_tt
    
    def pred_tt(self, graph_emb, no_pi):
        tt = []
        for pi in range(self.max_tt_len):
            tt.append(self.tt_pred[pi](graph_emb))
        tt = torch.tensor(tt).squeeze()
        return tt
    
    def pred_prob(self, hf):
        prob = self.prob_pred(hf)
        return prob