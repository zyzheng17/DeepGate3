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
from dg_datasets.dg3_multi_parser import OrderedData
import numpy as np
_transformer_factory = {
    'baseline': None,
    'plain': Plain_Transformer,
    'hop': Hop_Transformer, 
    'path': Path_Transformer
}

import torch.nn as nn
import time

def build_graph(g, area_nodes, area_nodes_stats, area_faninout_cone, prob):
    area_g = OrderedData()
    nodes = area_nodes[area_nodes != -1]
    pi_mask = (area_nodes_stats == 1)[:len(nodes)]
    area_g.nodes = nodes
    area_g.gate = g.gate[nodes]
    area_g.gate[pi_mask] = 0
    area_g.prob = prob[nodes]
    area_g.forward_level = g.forward_level[nodes]
    area_g.backward_level = g.backward_level[nodes]
    area_g.forward_index = torch.tensor(range(len(nodes)))
    area_g.backward_index = torch.tensor(range(len(nodes)))
    
    # Edge_index
    glo_to_area = {}
    for i, node in enumerate(nodes):
        glo_to_area[node.item()] = i
    area_edge_index = []
    for edge in g.edge_index.t():
        if edge[0].item() in glo_to_area and edge[1].item() in glo_to_area:
            area_edge_index.append([glo_to_area[edge[0].item()], glo_to_area[edge[1].item()]])
    area_edge_index = torch.tensor(area_edge_index).t()
    area_g.edge_index = area_edge_index
    
    area_g.fanin_fanout_cones = area_faninout_cone
    area_g.batch = torch.zeros(len(nodes), dtype=torch.long)
    return area_g

def merge_area_g(batch_g, g):
    no_nodes = batch_g.nodes.shape[0]
    batch_g.nodes = torch.cat([batch_g.nodes, g.nodes])
    batch_g.gate = torch.cat([batch_g.gate, g.gate])
    batch_g.prob = torch.cat([batch_g.prob, g.prob])
    batch_g.forward_level = torch.cat([batch_g.forward_level, g.forward_level])
    batch_g.backward_level = torch.cat([batch_g.backward_level, g.backward_level])
    batch_g.edge_index = torch.cat([batch_g.edge_index, g.edge_index + no_nodes], dim=1)
    batch_g.fanin_fanout_cones = torch.cat([batch_g.fanin_fanout_cones, g.fanin_fanout_cones], dim=0)
    batch_g.batch = torch.cat([batch_g.batch, torch.tensor([batch_g.batch.max() + 1] * len(g.nodes)).to(batch_g.batch.device)])
    
    batch_g.forward_index = torch.tensor(range(len(batch_g.nodes))).to(batch_g.batch.device)
    batch_g.backward_index = torch.tensor(range(len(batch_g.nodes))).to(batch_g.batch.device)
    
    return batch_g
    

class DeepGate3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_tt_len = 64
        self.hidden = 128
        self.max_path_len = 257
        self.tf_arch = args.tf_arch
        # Tokenizer
        self.tokenizer = DeepGate2()
        # Stone: Modify here
        if args.tf_arch != 'baseline':
            self.tokenizer.load_pretrained(args.pretrained_model_path)

        # for param in self.tokenizer.parameters():
        #     param.requires_grad = False

        #special token
        self.cls_token_hf = nn.Parameter(torch.randn([self.hidden,]))
        self.cls_token_hs = nn.Parameter(torch.randn([self.hidden,]))
        self.cls_path_token = nn.Parameter(torch.randn([self.hidden,]))
        self.dc_token = nn.Parameter(torch.randn([self.hidden,]))
        self.zero_token = nn.Parameter(torch.randn([self.hidden,]))
        self.one_token = nn.Parameter(torch.randn([self.hidden,]))
        self.pad_token = torch.zeros([self.hidden,]) # don't learn
        self.pool_max_length = 10
        self.hf_PositionalEmbedding = nn.Embedding(33,self.hidden)
        self.hs_PositionalEmbedding = nn.Embedding(33,self.hidden)
        self.Path_Pos = nn.Embedding(self.max_path_len,self.hidden)


        
        # Transformer 
        if args.tf_arch != 'baseline':
            self.transformer = _transformer_factory[args.tf_arch](args,hidden=self.hidden)
        

        #pooling layer
        pool_layer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=4, batch_first=True)
        self.hop_func_tf = nn.TransformerEncoder(pool_layer, num_layers=3)
        self.hop_struc_tf = nn.TransformerEncoder(pool_layer, num_layers=3)
        self.path_struc_tf = nn.TransformerEncoder(pool_layer, num_layers=3)
        
        

        # Prediction 
        self.hop_head = nn.Sequential(nn.Linear(self.hidden, self.hidden*4),
                        nn.ReLU(),
                        nn.LayerNorm(self.hidden*4),
                        nn.Linear(self.hidden*4, self.max_tt_len))
        self.readout_prob = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_len = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_and_ratio = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_hop_level = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
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
        self.on_hop_head = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_len = MLP(
            dim_in=self.args.token_emb, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )

        #Similarity
        self.proj_gate_ttsim = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_hop_ttsim = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_GED = MLP(
            dim_in=self.args.token_emb*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )

    
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
        
    def forward_large(self, g, max_area_batch=8):
        assert self.args.workload
        all_area_nodes = g.area_nodes
        all_area_nodes_stats = g.area_nodes_stats
        all_area_lev = g.area_lev
        all_area_faninout_cone = g.area_fanin_fanout_cone
        prob = g.prob.clone()
        glo_hs = torch.zeros([len(g.gate), self.hidden]).to(g.x.device)
        glo_hf = torch.zeros([len(g.gate), self.hidden]).to(g.x.device)

        # glo_hs, glo_hf = self.tokenizer(g, g.prob)
        
        batch_area_g_list = []
        curr_bs = 0
        for area_idx, area_nodes in enumerate(all_area_nodes):
            # if area_idx%10==0:
            #     print(f'encode area{area_idx}')
            area_nodes_stats = all_area_nodes_stats[area_idx]
            area_faninout_cone = all_area_faninout_cone[area_idx]
            area_g = build_graph(g, area_nodes, area_nodes_stats, area_faninout_cone, prob)

            area_g = area_g.to(g.x.device)
            hs, hf = self.tokenizer(area_g, area_g.prob)
            if self.tf_arch != 'baseline':
                hf_tf, hs_tf = self.transformer(area_g, hf, hs)
                #function
                hf = hf + hf_tf
                #structure
                hs = hs + hs_tf
            prob = self.readout_prob(hf).squeeze(-1)
            glo_hs[area_g.nodes] = hs.detach()
            glo_hf[area_g.nodes] = hf.detach()
            
        return glo_hs, glo_hf

        # for area_idx, area_nodes in enumerate(all_area_nodes):
        #     # if area_idx%10==0:
        #     #     print(f'encode area{area_idx}')
        #     area_nodes_stats = all_area_nodes_stats[area_idx]
        #     area_faninout_cone = all_area_faninout_cone[area_idx]
        #     area_g = build_graph(g, area_nodes, area_nodes_stats, area_faninout_cone, prob)
        #     # area_g = area_g
        #     if curr_bs == 0:
        #         batch_area_g = area_g.clone()
        #     else:
        #         batch_area_g = merge_area_g(batch_area_g, area_g)
        #     curr_bs = (curr_bs + 1) % max_area_batch
        #     if curr_bs == 0:
        #         batch_area_g_list.append(batch_area_g)
        # if curr_bs != 0:
        #     batch_area_g_list.append(batch_area_g)
        
        # for batch_idx, batch in enumerate(batch_area_g_list):
        #     batch = batch.to(g.x.device)
        #     hs, hf = self.tokenizer(batch, batch.prob)
        #     if self.tf_arch != 'baseline':
        #         hf_tf, hs_tf = self.transformer(batch, hf, hs)
        #         #function
        #         hf = hf + hf_tf
        #         #structure
        #         hs = hs + hs_tf
        #     prob = self.readout_prob(hf).squeeze(-1)
        #     glo_hs[batch.nodes] = hs.detach()
        #     glo_hf[batch.nodes] = hf.detach()
            
        # return glo_hs, glo_hf
        
    def forward(self, g, skip_path=False, skip_hop=False, large_ckt=False):
        if large_ckt:
            hs, hf = self.forward_large(g)
        else:
            # t = time.time()
            if self.args.workload:
                hs, hf = self.tokenizer(g, g.prob)
            else:
                hs, hf = self.tokenizer(g)

            # hf = hf.detach()
            # hs = hs.detach()

            # Refine-Transformer 

            if self.tf_arch != 'baseline':
                hf_tf, hs_tf = self.transformer(g, hf, hs)
                #function
                hf = hf + hf_tf
                #structure
                hs = hs + hs_tf

            # print('dg3 refine transformer time:{:.2f}'.format(time.time()-t))
            # t = time.time()

        #=========================================================
        #======================GATE-level=========================
        #=========================================================

        #gate-level pretrain task : predict pari-wise TT sim
        # max_len = min(g.tt_pair_index.shape[1],100000)
        # gate_tt_sim = self.proj_gate_ttsim(torch.cat([hf[g.tt_pair_index[0,:max_len]],hf[g.tt_pair_index[1,:max_len]]],dim=-1))
        # if g.tt_pair_index.shape[1] <= 1:
        #     print(g)
        gate_tt_sim = self.proj_gate_ttsim(torch.cat([hf[g.tt_pair_index[0]],hf[g.tt_pair_index[1]]],dim=-1))
        gate_tt_sim = nn.Sigmoid()(gate_tt_sim)

        #gate-level pretrain task : predict global probability
        prob = self.readout_prob(hf)

        #gate-level pretrain task : predict global level
        pred_level = self.readout_level(hs)

        #gate-level pretrain task : predict connection
        # max_len = min(g.connect_pair_index.shape[1],100000)
        gates = hs[g.connect_pair_index]
        gates = gates.permute(1,2,0).reshape(-1,self.hidden*2)
        pred_connect = self.connect_head(gates)

        # print('dg3 gate-level time:{:.2f}'.format(time.time()-t))
        # t = time.time()

        #=========================================================
        #======================PATH-level=========================
        #=========================================================
        if skip_path: 
            on_path_logits = 0
            pred_path_len = 0
            pred_path_and_ratio = 0
            # on_path_logits = torch.zeros(len(g.ninp_labels)).to(hs.device)
            # pred_path_len = torch.zeros(len(g.paths_len)).to(hs.device)
            # pred_path_and_ratio = torch.zeros(len(g.paths_and_ratio)).to(hs.device)
        else:
            # Stone: reduce maximum path length
            max_path_len = g.paths_len.max().item()     # previous: 256
            assert max_path_len < 128, "path length is too large"
            g.paths = g.paths[:,:max_path_len]
            
            if self.args.fast:
                tot_paths_cnt = len(g.paths)
                g_shuffle_index = np.arange(tot_paths_cnt)
                np.random.shuffle(g_shuffle_index)
                g_shuffle_index = g_shuffle_index[:int(tot_paths_cnt*0.1)]
                
                g.paths = g.paths[g_shuffle_index]
                g.path_forward_index = g.path_forward_index[:len(g_shuffle_index)]
                g.paths_len = g.paths_len[g_shuffle_index]
                g.paths_and_ratio = g.paths_and_ratio[g_shuffle_index]
                g.paths_no_and = g.paths_no_and[g_shuffle_index]
                g.paths_no_not = g.paths_no_not[g_shuffle_index]
                
                ninp_node_index = []
                ninp_path_index = []
                ninp_labels = []
                new_index_map = {}
                for k, idx in enumerate(g_shuffle_index):
                    new_index_map[idx] = k
                for k in range(len(g.ninp_node_index)):
                    if g.ninp_path_index[k].item() in g_shuffle_index:
                        ninp_node_index.append(g.ninp_node_index[k].item())
                        ninp_path_index.append(new_index_map[g.ninp_path_index[k].item()])
                        ninp_labels.append(g.ninp_labels[k].item())
                g.ninp_node_index = torch.tensor(ninp_node_index).to(g.paths.device)
                g.ninp_path_index = torch.tensor(ninp_path_index).to(g.paths.device)
                g.ninp_labels = torch.tensor(ninp_labels).to(g.paths.device)
                
            #path-level pretrain task : on-path prediction, path num prediction
            path_hs =  torch.cat([self.cls_path_token.reshape([1,1,-1]).repeat(g.paths.shape[0],1,1),hs[g.paths]],dim=1)
            # False = compute attention, True = mask 
            path_mask = torch.stack([torch.cat([torch.zeros([g.paths_len[i]+1]),torch.ones([max_path_len-g.paths_len[i]])],dim=0) for i in range(g.paths.shape[0])]) 
            path_mask = torch.where(path_mask==1, True, False).to(path_hs.device)
            pos = torch.arange(path_hs.shape[1]).unsqueeze(0).repeat(path_hs.shape[0],1).to(hs.device)
            path_hs = path_hs + self.Path_Pos(pos)
            path_hs = self.path_struc_tf(path_hs, src_key_padding_mask = path_mask)[:, 0]

            #on-path prediction
            on_path_emb = torch.cat([hs[g.ninp_node_index],path_hs[g.ninp_path_index]],dim=1)
            on_path_logits = self.on_path_head(on_path_emb)

            #path num prediction
            pred_path_len = self.readout_path_len(path_hs)
            pred_path_and_ratio = self.readout_path_and_ratio(path_hs)  # Predict the ratio of AND gates in the path

        # print('dg3 path-level time:{:.2f}'.format(time.time()-t))
        # t = time.time()
        #=========================================================
        #======================GRAPH-level========================
        #=========================================================
        if skip_hop:
            hop_tt = torch.zeros(len(g.hop_po), 64).to(hs.device)
            hop_tt_sim = torch.tensor(range(len(g.hop_pair_index[0])), dtype=float).to(hs.device)
            pred_hop_num = torch.zeros(len(g.hop_po)).to(hs.device)
            pred_hop_level = torch.zeros(len(g.hop_po)).to(hs.device)
            on_hop_logits = torch.zeros(len(g.ninh_node_index)).to(hs.device)
            pred_GED = torch.tensor(range(len(g.hop_pair_index[0])), dtype=float).to(hs.device)
        else:
            #graph-level pretrain task : predict truth table & pair-wise TT sim
            hop_hf = []
            hf_masks = []
            if self.tf_arch!='baseline':
                pi_emb = hf[g.hop_pi]
                pi_emb[g.hop_pi_stats==1] = self.one_token
                pi_emb[g.hop_pi_stats==0] = self.zero_token
                pi_emb[g.hop_pi_stats==-1] = self.dc_token 
                pi_emb[g.hop_pi_stats==-2] = self.pad_token.to(hf.device)
                po_emb = hf[g.hop_po]
                hop_hf = torch.cat([self.cls_token_hf.unsqueeze(0).unsqueeze(0).repeat(pi_emb.shape[0],1,1),pi_emb,po_emb], dim=1)
                hf_masks = torch.where(g.hop_pi_stats==-1,1,0)
                hf_masks = torch.cat([torch.zeros(g.hop_pi_stats.shape[0],1).to(hf.device),g.hop_pi_stats,torch.zeros(g.hop_pi_stats.shape[0],1).to(hf.device)],dim=1)
                
                # hop_hf = torch.stack(hop_hf) #bs seq_len hidden
                pos = torch.arange(hop_hf.shape[1]).unsqueeze(0).repeat(hop_hf.shape[0],1).to(hf.device)
                hop_hf = hop_hf + self.hf_PositionalEmbedding(pos)

                # hf_masks = 1 - torch.stack(hf_masks).to(hf.device).float() #bs seq_len 
                hf_masks = torch.where(hf_masks==1, True, False).to(hop_hf.device)
                hop_hf = self.hop_func_tf(hop_hf,src_key_padding_mask = hf_masks)
                hop_hf = hop_hf[:,0]
            else :
                hop_hf = torch.mean(torch.cat([pi_emb,po_emb],dim=1),dim=1)

            #pair-wise TT sim prediction
            hop_tt_sim = self.proj_hop_ttsim(torch.cat([hop_hf[g.hop_forward_index[g.hop_pair_index[0]]],hop_hf[g.hop_forward_index[g.hop_pair_index[1]]]],dim=-1))
            hop_tt_sim = nn.Sigmoid()(hop_tt_sim)

            # truth table prediction
            hop_tt = self.hop_head(hop_hf)

            # hop_hs = torch.stack(hop_hs) #bs seq_len hidden
            if self.tf_arch!='baseline':
                hop_hs = hs[g.hop_nodes]
                hop_hs = torch.cat([self.cls_token_hs.reshape(1,1,-1).repeat(hop_hs.shape[0],1,1),hop_hs],dim=1)
                
                pos = torch.arange(hop_hs.shape[1]).unsqueeze(0).repeat(hop_hs.shape[0],1).to(hs.device)
                hop_hs = hop_hs + self.hs_PositionalEmbedding(pos)

                # hs_masks = 1 - torch.stack(hs_masks).to(hs.device).float() #bs seq_len 
                hs_masks = 1 - g.hop_nodes_stats
                hs_masks = torch.cat([torch.zeros([hs_masks.shape[0],1]).to(hs.device),hs_masks],dim=1)
                hs_masks = torch.where(hs_masks==1, True, False).to(hop_hs.device)

                hop_hs = self.hop_struc_tf(hop_hs,src_key_padding_mask = hs_masks)
                hop_hs = hop_hs[:,0]

            hop_hs = torch.mean(hs[g.hop_nodes],dim=1)
            #pari-wise GED prediction 
            pred_GED = self.proj_GED(torch.cat([hop_hs[g.hop_pair_index[0]],hop_hs[g.hop_pair_index[1]]],dim=-1))
            pred_GED = nn.Sigmoid()(pred_GED)

            #gate number prediction
            pred_hop_num = self.readout_num(hop_hs)

            #hop level prediction
            pred_hop_level = self.readout_hop_level(hop_hs)

            #graph-level pretrain task: on hop prediction
            on_hop_emb = torch.cat([hs[g.ninh_node_index],hop_hs[g.ninh_hop_index]],dim=1)
            on_hop_logits = self.on_hop_head(on_hop_emb)


        # print('dg3 graph-level time:{:.2f}'.format(time.time()-t))
        # t = time.time()
        
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
                'tt_sim':gate_tt_sim,
            },
            'path':{
                'on_path':on_path_logits,
                'length':pred_path_len,
                'AND': pred_path_and_ratio, 
            },
            'hop':{
                'tt':hop_tt,
                'tt_sim':hop_tt_sim,
                'area':pred_hop_num,
                'time':pred_hop_level,
                'on_hop':on_hop_logits,
                'GED':pred_GED,
            }
        }
        return result


    def test_large(self, g, skip_path=False, skip_hop=False):
        max_lv = torch.max(g.area_lev)
        for l in range(max_lv):
            if l not in g.area_lev:
                continue
            hop_l = g.area_nodes[g.area_lev==l]
            ff_cone = g.area_fanin_fanout_cone[g.area_lev==l]
            

        # t = time.time()
        if self.args.workload:
            hs, hf = self.tokenizer(g, g.prob)
        else:
            hs, hf = self.tokenizer(g)
        hf = hf.detach()
        hs = hs.detach()

        # Refine-Transformer 

        if self.tf_arch != 'baseline':

            hf_tf, hs_tf = self.transformer(g, hf, hs)
            #function
            hf = hf + hf_tf
            #structure
            hs = hs + hs_tf

        # print('dg3 refine transformer time:{:.2f}'.format(time.time()-t))
        # t = time.time()

        #=========================================================
        #======================GATE-level=========================
        #=========================================================
            
        #gate-level pretrain task : predict pari-wise TT sim
        gate_tt_sim = self.proj_gate_ttsim(torch.cat([hf[g.tt_pair_index[0]],hf[g.tt_pair_index[1]]],dim=-1))
        gate_tt_sim = nn.Sigmoid()(gate_tt_sim)

        #gate-level pretrain task : predict global probability
        prob = self.readout_prob(hf)

        #gate-level pretrain task : predict global level
        pred_level = self.readout_level(hs)

        #gate-level pretrain task : predict connection
        gates = hs[g.connect_pair_index]
        gates = gates.permute(1,2,0).reshape(-1,self.hidden*2)
        pred_connect = self.connect_head(gates)

        # print('dg3 gate-level time:{:.2f}'.format(time.time()-t))
        # t = time.time()

        #=========================================================
        #======================PATH-level=========================
        #=========================================================
        if skip_path: 
            on_path_logits = 0
            pred_path_len = 0
            pred_path_and_ratio = 0
            # on_path_logits = torch.zeros(len(g.ninp_labels)).to(hs.device)
            # pred_path_len = torch.zeros(len(g.paths_len)).to(hs.device)
            # pred_path_and_ratio = torch.zeros(len(g.paths_and_ratio)).to(hs.device)
        else:
            # Stone: reduce maximum path length
            max_path_len = g.paths_len.max().item()     # previous: 256
            assert max_path_len < 128, "path length is too large"
            g.paths = g.paths[:,:max_path_len]
            
            if self.args.fast:
                tot_paths_cnt = len(g.paths)
                g_shuffle_index = np.arange(tot_paths_cnt)
                np.random.shuffle(g_shuffle_index)
                g_shuffle_index = g_shuffle_index[:int(tot_paths_cnt*0.1)]
                
                g.paths = g.paths[g_shuffle_index]
                g.path_forward_index = g.path_forward_index[:len(g_shuffle_index)]
                g.paths_len = g.paths_len[g_shuffle_index]
                g.paths_and_ratio = g.paths_and_ratio[g_shuffle_index]
                g.paths_no_and = g.paths_no_and[g_shuffle_index]
                g.paths_no_not = g.paths_no_not[g_shuffle_index]
                
                ninp_node_index = []
                ninp_path_index = []
                ninp_labels = []
                new_index_map = {}
                for k, idx in enumerate(g_shuffle_index):
                    new_index_map[idx] = k
                for k in range(len(g.ninp_node_index)):
                    if g.ninp_path_index[k].item() in g_shuffle_index:
                        ninp_node_index.append(g.ninp_node_index[k].item())
                        ninp_path_index.append(new_index_map[g.ninp_path_index[k].item()])
                        ninp_labels.append(g.ninp_labels[k].item())
                g.ninp_node_index = torch.tensor(ninp_node_index).to(g.paths.device)
                g.ninp_path_index = torch.tensor(ninp_path_index).to(g.paths.device)
                g.ninp_labels = torch.tensor(ninp_labels).to(g.paths.device)
                
            #path-level pretrain task : on-path prediction, path num prediction
            path_hs =  torch.cat([self.cls_path_token.reshape([1,1,-1]).repeat(g.paths.shape[0],1,1),hs[g.paths]],dim=1)
            # False = compute attention, True = mask 
            path_mask = torch.stack([torch.cat([torch.zeros([g.paths_len[i]+1]),torch.ones([max_path_len-g.paths_len[i]])],dim=0) for i in range(g.paths.shape[0])]) 
            path_mask = torch.where(path_mask==1, True, False).to(path_hs.device)
            pos = torch.arange(path_hs.shape[1]).unsqueeze(0).repeat(path_hs.shape[0],1).to(hs.device)
            path_hs = path_hs + self.Path_Pos(pos)
            path_hs = self.path_struc_tf(path_hs, src_key_padding_mask = path_mask)[:, 0]

            #on-path prediction
            on_path_emb = torch.cat([hs[g.ninp_node_index],path_hs[g.ninp_path_index]],dim=1)
            on_path_logits = self.on_path_head(on_path_emb)

            #path num prediction
            pred_path_len = self.readout_path_len(path_hs)
            pred_path_and_ratio = self.readout_path_and_ratio(path_hs)  # Predict the ratio of AND gates in the path

        # print('dg3 path-level time:{:.2f}'.format(time.time()-t))
        # t = time.time()
        #=========================================================
        #======================GRAPH-level========================
        #=========================================================
        if skip_hop:
            hop_tt = torch.zeros(len(g.hop_po), 64).to(hs.device)
            hop_tt_sim = torch.tensor(range(len(g.hop_pair_index[0])), dtype=float).to(hs.device)
            pred_hop_num = torch.zeros(len(g.hop_po)).to(hs.device)
            pred_hop_level = torch.zeros(len(g.hop_po)).to(hs.device)
            on_hop_logits = torch.zeros(len(g.ninh_node_index)).to(hs.device)
            pred_GED = torch.tensor(range(len(g.hop_pair_index[0])), dtype=float).to(hs.device)
        else:
            #graph-level pretrain task : predict truth table & pair-wise TT sim
            hop_hf = []
            hf_masks = []
            
            pi_emb = hf[g.hop_pi]
            pi_emb[g.hop_pi_stats==1] = self.one_token
            pi_emb[g.hop_pi_stats==0] = self.zero_token
            pi_emb[g.hop_pi_stats==-1] = self.dc_token 
            pi_emb[g.hop_pi_stats==-2] = self.pad_token.to(hf.device)
            po_emb = hf[g.hop_po]
            hop_hf = torch.cat([self.cls_token_hf.unsqueeze(0).unsqueeze(0).repeat(pi_emb.shape[0],1,1),pi_emb,po_emb], dim=1)
            hf_masks = torch.where(g.hop_pi_stats==-1,1,0)
            hf_masks = torch.cat([torch.zeros(g.hop_pi_stats.shape[0],1).to(hf.device),g.hop_pi_stats,torch.zeros(g.hop_pi_stats.shape[0],1).to(hf.device)],dim=1)
            
            # hop_hf = torch.stack(hop_hf) #bs seq_len hidden
            pos = torch.arange(hop_hf.shape[1]).unsqueeze(0).repeat(hop_hf.shape[0],1).to(hf.device)
            hop_hf = hop_hf + self.hf_PositionalEmbedding(pos)

            # hf_masks = 1 - torch.stack(hf_masks).to(hf.device).float() #bs seq_len 
            hf_masks = torch.where(hf_masks==1, True, False).to(hop_hf.device)
            hop_hf = self.hop_func_tf(hop_hf,src_key_padding_mask = hf_masks)
            hop_hf = hop_hf[:,0]

            #pair-wise TT sim prediction
            hop_tt_sim = self.proj_hop_ttsim(torch.cat([hop_hf[g.hop_forward_index[g.hop_pair_index[0]]],hop_hf[g.hop_forward_index[g.hop_pair_index[1]]]],dim=-1))
            hop_tt_sim = nn.Sigmoid()(hop_tt_sim)

            # truth table prediction
            hop_tt = self.hop_head(hop_hf)

            # hop_hs = torch.stack(hop_hs) #bs seq_len hidden
            hop_hs = hs[g.hop_nodes]
            hop_hs = torch.cat([self.cls_token_hs.reshape(1,1,-1).repeat(hop_hs.shape[0],1,1),hop_hs],dim=1)
            
            pos = torch.arange(hop_hs.shape[1]).unsqueeze(0).repeat(hop_hs.shape[0],1).to(hs.device)
            hop_hs = hop_hs + self.hs_PositionalEmbedding(pos)

            # hs_masks = 1 - torch.stack(hs_masks).to(hs.device).float() #bs seq_len 
            hs_masks = 1 - g.hop_nodes_stats
            hs_masks = torch.cat([torch.zeros([hs_masks.shape[0],1]).to(hs.device),hs_masks],dim=1)
            hs_masks = torch.where(hs_masks==1, True, False).to(hop_hs.device)

            hop_hs = self.hop_struc_tf(hop_hs,src_key_padding_mask = hs_masks)
            hop_hs = hop_hs[:,0]

            #pari-wise GED prediction 
            pred_GED = self.proj_GED(torch.cat([hop_hs[g.hop_pair_index[0]],hop_hs[g.hop_pair_index[1]]],dim=-1))
            pred_GED = nn.Sigmoid()(pred_GED)

            #gate number prediction
            pred_hop_num = self.readout_num(hop_hs)

            #hop level prediction
            pred_hop_level = self.readout_hop_level(hop_hs)

            #graph-level pretrain task: on hop prediction
            on_hop_emb = torch.cat([hs[g.ninh_node_index],hop_hs[g.ninh_hop_index]],dim=1)
            on_hop_logits = self.on_hop_head(on_hop_emb)


        # print('dg3 graph-level time:{:.2f}'.format(time.time()-t))
        # t = time.time()
        
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
                'tt_sim':gate_tt_sim,
            },
            'path':{
                'on_path':on_path_logits,
                'length':pred_path_len,
                'AND': pred_path_and_ratio, 
            },
            'hop':{
                'tt':hop_tt,
                'tt_sim':hop_tt_sim,
                'area':pred_hop_num,
                'time':pred_hop_level,
                'on_hop':on_hop_logits,
                'GED':pred_GED,
            }
        }
        return result
   
