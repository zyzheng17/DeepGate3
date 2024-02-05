import deepgate as dg 
import numpy as np 
import torch
from torch_geometric.data import Data

def npzitem_to_graph(cir_name, x_data, edge_index, tt):
    x_data = np.array(x_data)
    edge_index = np.array(edge_index)
    tt_dis = [-1]
    tt_pair_index = [[-1, -1]]
    prob = [0] * len(x_data)
    rc_pair_index = [[-1, -1]]
    is_rc = [-1]
    graph = dg.parse_pyg_mlpgate(
        x_data, edge_index, tt_dis, tt_pair_index, prob, rc_pair_index, is_rc
    )
    graph.name = cir_name
    graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
    graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
    graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
    graph.tt = tt
        
    return graph

class OrderedData(Data):
    def __init__(self, edge_index=None, x=None, y=None, \
                 forward_level=None, forward_index=None, backward_level=None, backward_index=None, \
                 PIs=None, POs=None, tt=None):
        super().__init__()
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index
        self.PIs = PIs
        self.POs = POs
        self.tt = tt
        # self.tt_pair_index = [[-1, -1]]
        # self.rc_pair_index = [[-1, -1]]
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key or 'PIs' in key or 'POs' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index" or key == 'tt_pair_index' or key == 'rc_pair_index':
            return 1
        else:
            return 0
        
def topological_sort(x, fanout_list):
    visited = [False] * len(x)
    order = []
    
    def dfs(u):
        visited[u] = True
        for v in fanout_list[u]:
            if not visited[v]:
                dfs(v)
        order.insert(0, u)
    
    for i in range(len(x)):
        if not visited[i]:
            dfs(i)
    
    return order

def parse_pyg_dg3(x, edge_index, tt, 
                      num_gate_types=3):
    x_torch = dg.construct_node_feature(x, num_gate_types)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Get topological order
    fanin_list = []
    fanout_list = []
    for i in range(len(x)):
        fanin_list.append([])
        fanout_list.append([])
    for edge in edge_index:
        fanin_list[edge[1].item()].append(edge[0].item())
        fanout_list[edge[0].item()].append(edge[1].item())
    order = topological_sort(x, fanout_list)
    order = torch.tensor(order, dtype=torch.long)
    assert len(order) == len(x)
        
    if len(edge_index) == 0:
        edge_index = edge_index.t().contiguous()
        forward_index = torch.LongTensor([i for i in range(len(x))])
        backward_index = torch.LongTensor([i for i in range(len(x))])
        forward_level = torch.zeros(len(x))
        backward_level = torch.zeros(len(x))
    else:
        edge_index = edge_index.t().contiguous()
        forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, x_torch.size(0))

    PIs = forward_index[(forward_level == 0) & (backward_level != 0)]
    POs = backward_index[(backward_level == 0) & (forward_level != 0)]
    
    graph = OrderedData(x=x_torch, edge_index=edge_index, 
                        forward_level=forward_level, forward_index=forward_index, 
                        backward_level=backward_level, backward_index=backward_index, 
                        tt=tt, PIs=PIs, POs=POs)
    # tt_dis = [-1]
    # tt_pair_index = [[-1, -1]]
    # prob = [0] * len(x_data)
    # rc_pair_index = [[-1, -1]]
    # is_rc = [-1]
    # graph = OrderedData(x=x_torch, edge_index=edge_index, 
    #                     rc_pair_index=rc_pair_index, is_rc=is_rc,
    #                     tt_pair_index=tt_pair_index, tt_dis=tt_dis, 
    #                     forward_level=forward_level, forward_index=forward_index, 
    #                     backward_level=backward_level, backward_index=backward_index,
    #                     tt=tt, PIs=PIs, POs=POs)
    graph.use_edge_attr = False

    # add reconvegence info
    # graph.rec = torch.tensor(x[:, 3:4], dtype=torch.float)
    # graph.rec_src = torch.tensor(x[:, 4:5], dtype=torch.float)
    # add gt info
    # add indices for gate types
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
    graph.order = order

    return graph