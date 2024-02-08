import torch
import copy

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
    hop_nodes += target_idx
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
        
    return hop_nodes
    
    