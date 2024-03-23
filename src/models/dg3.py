import torch 
import deepgate as dg
import torch.nn as nn 

from .mlp import MLP
from .dg2 import DeepGate2

from .plain_tf import Plain_Transformer
from .hop_tf import Hop_Transformer
from .path_tf import Path_Transformer
from .baseline_tf import Baseline_Transformer
from .mlp import MLP
from .tf_pool import tf_Pooling
import numpy as np
_transformer_factory = {
    'baseline': None,
    'plain': Plain_Transformer,
    'hop': Hop_Transformer, 
    'path': Path_Transformer
}

import torch.nn as nn

class DeepGate3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_tt_len = 64
        self.hidden = 128
        self.max_path_len = 64
        self.tf_arch = args.tf_arch
        # Tokenizer
        self.tokenizer = DeepGate2()
        self.tokenizer.load_pretrained(args.pretrained_model_path)

        #special token
        self.cls_token = nn.Parameter(torch.randn([self.hidden,]))
        self.cls_path_token = nn.Parameter(torch.randn([self.hidden,]))
        self.dc_token = nn.Parameter(torch.randn([self.hidden,]))
        self.zero_token = nn.Parameter(torch.randn([self.hidden,]))
        self.one_token = nn.Parameter(torch.randn([self.hidden,]))
        self.pad_token = torch.zeros([self.hidden,]) # dont learn
        self.pool_max_length = 10
        self.PositionalEmbedding = nn.Embedding(10,self.hidden)
        self.Path_Pos = nn.Embedding(self.max_path_len,self.hidden)
        
        # Transformer 
        if args.tf_arch != 'baseline':
            self.transformer = _transformer_factory[args.tf_arch](args)
        
        # Prediction 
        self.readout_prob = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_gate1 = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_gate2 = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_len = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_hop_level = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        #pooling layer
        pool_layer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=4, batch_first=True)
        self.tf_encoder = nn.TransformerEncoder(pool_layer, num_layers=3)
        self.hop_head = nn.Sequential(nn.Linear(self.hidden, self.hidden*4),
                        nn.ReLU(),
                        nn.LayerNorm(self.hidden*4),
                        nn.Linear(self.hidden*4, self.max_tt_len))
        self.pathformer = nn.TransformerEncoder(pool_layer, num_layers=3)

        # Prediction 
        self.readout_level = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_num = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.connect_head = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=3, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.on_path_head = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_len = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_gate1 = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_gate2 = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        
        # function & structure head

    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        
        


    def forward(self, g):
        hs, hf = self.tokenizer(g)
        hf = hf.detach()
        hs = hs.detach()
        # Refine-Transformer 
        if self.tf_arch != 'baseline':

            hf_tf, hs_tf = self.transformer(g, hs, hf)
            #function
            hf = hf + hf_tf
            #structure
            hs = hs + hs_tf
            # hs = hs + self.structure_head(h)
        #===================function======================
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

        masks = 1 - torch.stack(masks).to(hf.device).float() #bs seq_len 
        
        hop_hf = self.tf_encoder(hop_hf,src_key_padding_mask = masks.float())
        hop_hf = hop_hf[:,0]
        hop_tt = self.hop_head(hop_hf)

        #===================strucutre======================
        #gate-level pretrain task : predict global level
        pred_level = self.readout_level(hs)

        #gate-level pretrain task : predict connection
        # src_gate,tgt_gate = self.get_gate_pair(g)
        gates = hs[g.connect_pair_index]
        gates = gates.permute(1,2,0).reshape(-1,self.hidden*2)
        pred_connect = self.connect_head(gates)

        #path-level pretrain task : on-path prediction, path num prediction
        #get path
        path_hs = []
        gate_1 = []
        gate_2 = []
        on_path_gates = []
        out_path_gates = []
        for i in range(g.paths.shape[0]):
            path_idx = g.paths[i][:g.paths_len[i]]
            gate_1.append((g.gate[path_idx]==1).sum())
            gate_2.append((g.gate[path_idx]==2).sum())
            #random select on-path gate
            on_path_gate = path_idx[torch.randint(0,path_idx.shape[0],[1])]
            on_path_gates.append(on_path_gate)
            #random select out-path gate
            gate_candidate = torch.argwhere(g.batch==g.batch[path_idx[0]])
            out_path_gate = gate_candidate[torch.randint(0,gate_candidate.shape[0],[1])]
            while out_path_gate in path_idx:
                out_path_gate = gate_candidate[torch.randint(0,gate_candidate.shape[0],[1])]
            out_path_gates.append(out_path_gate)

            path_emb = hs[path_idx]
            path_emb = torch.cat([self.cls_path_token.unsqueeze(0), path_emb,torch.zeros([self.max_path_len-1-path_emb.shape[0],self.hidden]).to(hf.device)],dim=0)
            path_hs.append(path_emb)
        on_path_gates = torch.stack(on_path_gates)
        out_path_gates = torch.stack(out_path_gates)
        path_hs = torch.stack(path_hs)
        gate_1 = torch.tensor(gate_1)
        gate_2 = torch.tensor(gate_2)
        pos = torch.arange(path_hs.shape[1]).unsqueeze(0).repeat(path_hs.shape[0],1).to(hs.device)
        path_hs = path_hs + self.Path_Pos(pos)
        path_hs = self.pathformer(path_hs)[:, 0]

        #on-path prediction
        on_path_emb = hs[on_path_gates].squeeze(1)
        out_path_emb = hs[out_path_gates].squeeze(1)
        on_path_emb = torch.cat([on_path_emb,path_hs],dim=-1)
        out_path_emb = torch.cat([out_path_emb,path_hs],dim=-1)
        on_path_logits = self.on_path_head(on_path_emb)
        out_path_logits = self.on_path_head(out_path_emb)
        #path num prediction
        pred_path_len = self.readout_path_len(path_hs)
        pred_path_gate1 = self.readout_gate1(path_hs)
        pred_path_gate2 = self.readout_gate2(path_hs)

        #graph-level pretrain task : predict truth table
        hop_hs = []
        masks = []
        for i in range(g.hop_po.shape[0]):
            pi_idx = g.hop_pi[i][g.hop_pi_stats[i]!=-1].squeeze(-1)
            pi_hop_stats = g.hop_pi_stats[i]
            pi_emb = hs[pi_idx]
            pi_emb = []
            for j in range(8):
                if pi_hop_stats[j] == -1:
                    continue
                else:
                    pi_emb.append(hs[g.hop_pi[i][j]])
            # pad seq to fixed length
            mask = [1 for _ in range(len(pi_emb))]
            while len(pi_emb) < 8:
                pi_emb.append(self.pad_token.to(hs.device))
                mask.append(0)

            pi_emb = torch.stack(pi_emb) # 8 128
            po_emb = hs[g.hop_po[i]] # 1 128
            hop_hs.append(torch.cat([self.cls_token.unsqueeze(0),pi_emb,po_emb], dim=0)) 
            mask.insert(0,1)
            mask.append(1)
            masks.append(torch.tensor(mask))

        hop_hs = torch.stack(hop_hs) #bs seq_len hidden
        pos = torch.arange(hop_hs.shape[1]).unsqueeze(0).repeat(hop_hs.shape[0],1).to(hs.device)
        hop_hs = hop_hs + self.PositionalEmbedding(pos)

        masks = 1 - torch.stack(masks).to(hs.device).float() #bs seq_len 
        
        hop_hs = self.tf_encoder(hop_hs,src_key_padding_mask = masks.float())
        hop_hs = hop_hs[:,0]
        pred_hop_num = self.readout_num(hop_hs)
        pred_hop_level = self.readout_hop_level(hop_hs)
        
        result = {
            'emb':
            {
                'hf':hf,
                'hs':hs,
            },
            'node':
            {
                'prob':prob,
                'level':pred_level,
                'connect':pred_connect,
            },
            'path':{
                'on_path':on_path_logits,
                'out_path':out_path_logits,
                'length':pred_path_len,
                'AND': pred_path_gate1,
                'NOT': pred_path_gate2
            },
            'hop':{
                'tt':hop_tt,
                'area':pred_hop_num,
                'time':pred_hop_level,
            }
        }
        return result
    

class DeepGate3_structure(nn.Module):
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
        self.readout_level = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_num = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.cls_head = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=3, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        

        #pooling layer
        pool_layer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=4, batch_first=True)
        self.tf_encoder = nn.TransformerEncoder(pool_layer, num_layers=3)
        

        
    def get_gate_pair(self, g):
        bs = g.batch.max().item() + 1
        src_per_batch = 256
        src_list = []
        gate0_list = []
        gate1_list = []
        gate2_list = []
        for i in range(bs):
            #delete PI and PO
            # all_src = torch.argwhere(g.batch==i).squeeze().cpu().numpy()
            # PI_idx = torch.argwhere(g.forward_level[all_src]==0).squeeze().cpu().numpy()
            # PO_idx = torch.argwhere(g.backward_level[all_src]==0).squeeze().cpu().numpy()
            all_src = g.forward_index[g.batch==i].squeeze()
            PI_idx = all_src[g.forward_level[all_src]==0].squeeze().cpu().numpy()
            PO_idx = all_src[g.backward_level[all_src]==0].squeeze().cpu().numpy()
            all_src = np.setdiff1d(all_src, PI_idx, assume_unique=False) 
            all_src = np.setdiff1d(all_src, PO_idx, assume_unique=False) 
            all_src = torch.tensor(all_src)
            
            #random choose source gate
            src_idx = torch.randint(0,all_src.shape[0],[src_per_batch])
            src = all_src[src_idx]
            src_list.append(src)
            # get target gate
            # all_tgt = torch.argwhere(g.batch==i).squeeze()
            all_tgt = g.forward_index[g.batch==i].squeeze()
            label =  g.fanin_fanout_cone[src]
            for j in range(src_per_batch):
                #each class get 1 data point to make it balance
                #class 0: no connection
                # gate_0 = torch.argwhere(label[j]==0).squeeze(-1)
                gate_0 = all_src[(label[j]==0)[:len(all_src)]].squeeze(-1)
                gate_0 = gate_0[torch.randint(0,gate_0.shape[0],[1])]
                gate0_list.append(all_tgt[gate_0])
                #class 1: message in 
                # gate_1 = torch.argwhere(label[j]==1).squeeze(-1)
                gate_1 = all_src[(label[j]==1)[:len(all_src)]].squeeze(-1)
                gate_1 = gate_1[torch.randint(0,gate_1.shape[0],[1])]
                gate1_list.append(all_tgt[gate_1])
                #class 2: message out
                # gate_2 = torch.argwhere(label[j]==2).squeeze(-1)
                gate_2 = all_src[(label[j]==2)[:len(all_src)]].squeeze(-1)
                gate_2 = gate_2[torch.randint(0,gate_2.shape[0],[1])]
                gate2_list.append(all_tgt[gate_2])

        src_list = torch.stack(src_list)
        gate0_list = torch.stack(gate0_list)
        gate1_list = torch.stack(gate1_list)
        gate2_list = torch.stack(gate2_list)

        
        
    def forward(self, g):
        bs = g.batch.max().item() + 1

        hs, hf = self.tokenizer(g)
        hf = hf.detach()
        hs = hs.detach()
        
        # Refine-Transformer 
        if self.tf_arch != 'baseline':
            #non-residual
            hs = self.transformer(g, hs, hf)

            #with-residual
            # hf = hf + self.transformer(g, hs, hf)

        #gate-level pretrain task : predict global level
        pred_level = self.readout_level(hs)

        #gate-level pretrain task : predict connection
        # src_gate,tgt_gate = self.get_gate_pair(g)
        gates = hs[g.connect_pair_index]
        gates = gates.permute(1,2,0).reshape(-1,self.hidden*2)
        pred_connect = self.cls_head(gates)

        #graph-level pretrain task : predict truth table
        hop_hs = []
        masks = []

        for i in range(g.hop_po.shape[0]):
            pi_idx = g.hop_pi[i][g.hop_pi_stats[i]!=-1].squeeze(-1)
            pi_hop_stats = g.hop_pi_stats[i]
            pi_emb = hs[pi_idx]
            pi_emb = []
            for j in range(8):
                if pi_hop_stats[j] == -1:
                    continue
                else:
                    pi_emb.append(hs[g.hop_pi[i][j]])
            # pad seq to fixed length
            mask = [1 for _ in range(len(pi_emb))]
            while len(pi_emb) < 8:
                pi_emb.append(self.pad_token.to(hs.device))
                mask.append(0)

            pi_emb = torch.stack(pi_emb) # 8 128
            po_emb = hs[g.hop_po[i]] # 1 128
            hop_hs.append(torch.cat([self.cls_token.unsqueeze(0),pi_emb,po_emb], dim=0)) 
            mask.insert(0,1)
            mask.append(1)
            masks.append(torch.tensor(mask))

        hop_hs = torch.stack(hop_hs) #bs seq_len hidden
        pos = torch.arange(hop_hs.shape[1]).unsqueeze(0).repeat(hop_hs.shape[0],1).to(hs.device)
        hop_hs = hop_hs + self.PositionalEmbedding(pos)

        masks = 1 - torch.stack(masks).to(hs.device).float() #bs seq_len 
        
        hop_hs = self.tf_encoder(hop_hs,src_key_padding_mask = masks.float())
        hop_hs = hop_hs[:,0]
        pred_hop_num = self.readout_num(hop_hs)
        
        return hs, hf, pred_level, pred_connect, pred_hop_num