import deepgate as dg 
import numpy as np 

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