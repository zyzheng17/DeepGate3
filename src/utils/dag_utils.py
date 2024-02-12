import torch
import copy
import networkx as nx
import numpy as np 
import deepgate as dg

def subgraph(target_idx, edge_index, edge_attr=None, dim=0):
    '''
    function from DAGNN
    '''
    le_idx = []
    for n in target_idx:
        ne_idx = edge_index[dim] == n
        le_idx += [ne_idx.nonzero().squeeze(-1)]
    le_idx = torch.cat(le_idx, dim=-1)
    lp_edge_index = edge_index[:, le_idx]
    if edge_attr is not None:
        lp_edge_attr = edge_attr[le_idx, :]
    else:
        lp_edge_attr = None
    return lp_edge_index, lp_edge_attr

def subgraph_hop(target_idx, edge_index, hops=3, dim=1):
    last_target_idx = copy.deepcopy(target_idx)
    curr_target_idx = []
    hop_nodes = []
    for k_hops in range(hops):
        if len(last_target_idx) == 0:
            break
        for n in last_target_idx:
            ne_idx = edge_index[dim] == n
            curr_target_idx += edge_index[1-dim, ne_idx].tolist()
            hop_nodes += edge_index[1-dim, ne_idx].unique().tolist()
        last_target_idx = list(set(curr_target_idx))
        curr_target_idx = []
        
    hop_nodes = torch.tensor(hop_nodes).unique()
    pis = torch.tensor(last_target_idx).unique()
        
    return hop_nodes, pis
    
def get_all_hops(g, hops=3): 
    subgraph = {}
    for idx in range(len(g.gate)):
        last_target_idx = copy.deepcopy([idx])
        curr_target_idx = []
        hop_nodes = []
        hop_edges = torch.zeros((2, 0), dtype=torch.long)
        for k_hops in range(hops):
            if len(last_target_idx) == 0:
                break
            for n in last_target_idx:
                ne_mask = g.edge_index[1] == n
                curr_target_idx += g.edge_index[0, ne_mask].tolist()
                hop_edges = torch.cat([hop_edges, g.edge_index[:, ne_mask]], dim=-1)
                hop_nodes += g.edge_index[0, ne_mask].unique().tolist()
            last_target_idx = list(set(curr_target_idx))
            curr_target_idx = []
        
        if len(hop_nodes) < 2:
            continue
        hop_nodes = torch.tensor(hop_nodes).unique().long()
        hop_nodes = torch.cat([hop_nodes, torch.tensor([idx])])
        hop_gates = g.gate[hop_nodes]
        
        # logic level 
        index_m = {}
        index_m_r = {}
        for k in hop_nodes:
            new_k = len(index_m.keys())
            index_m[k.item()] = new_k
            index_m_r[new_k] = k.item()
        new_edge_index = hop_edges.clone()
        for k in range(len(new_edge_index[0])):
            new_edge_index[0][k] = index_m[new_edge_index[0][k].item()]
            new_edge_index[1][k] = index_m[new_edge_index[1][k].item()]
        forward_level, forward_index, backward_level, _ = dg.return_order_info(new_edge_index, len(hop_nodes))
        subgraph[idx] = {}
        subgraph[idx]['nodes'] = hop_nodes
        subgraph[idx]['edges'] = hop_edges
        subgraph[idx]['gates'] = hop_gates
        subgraph[idx]['forward_level'] = forward_level
        subgraph[idx]['backward_level'] = backward_level
        subgraph[idx]['pis'] = hop_nodes[(forward_level==0) & (backward_level!=0)]
        subgraph[idx]['pos'] = hop_nodes[(forward_level!=0) & (backward_level==0)]
        
    return subgraph


        