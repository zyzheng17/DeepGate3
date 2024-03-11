import random 
import torch
import os 
import numpy as np
import networkx as nx
from utils.utils import run_command, hash_arr

def logic(gate_type, signals):
    if gate_type == 1:  # AND
        for s in signals:
            if s == 0:
                return 0
        return 1

    elif gate_type == 2:  # NOT
        for s in signals:
            if s == 1:
                return 0
            else:
                return 1

def complete_simulation(g_pis, g_pos, g_forward_level, g_nodes, g_edges, g_gates, pi_stats=[]):
    no_pi = len(g_pis) - sum([1 for x in pi_stats if x != 2])
    level_list = []
    fanin_list = []
    index_m = {}
    for level in range(g_forward_level.max()+1):
        level_list.append([])
    for k, idx in enumerate(g_nodes):
        level_list[g_forward_level[k].item()].append(k)
        fanin_list.append([])
        index_m[idx.item()] = k
    for edge in g_edges.t():
        fanin_list[index_m[edge[1].item()]].append(index_m[edge[0].item()])
    
    states = [-1] * len(g_nodes)
    po_tt = []
    for pattern_idx in range(2**no_pi):
        pattern = [int(x) for x in list(bin(pattern_idx)[2:].zfill(no_pi))]
        k = 0 
        while k < len(pi_stats):
            pi_idx = g_pis[k].item()
            if pi_stats[k] == 2:
                states[index_m[pi_idx]] = pattern[k]
            elif pi_stats[k] == 1:
                states[index_m[pi_idx]] = 1
            elif pi_stats[k] == 0:
                states[index_m[pi_idx]] = 0
            else:
                raise ValueError('Invalid pi_stats')
            k += 1
        for level in range(1, len(level_list), 1):
            for node_k in level_list[level]:
                source_signals = []
                for pre_k in fanin_list[node_k]:
                    source_signals.append(states[pre_k])
                if len(source_signals) == 0:
                    continue
                states[node_k] = logic(g_gates[node_k].item(), source_signals)
        po_tt.append(states[index_m[g_pos.item()]])
    
    return po_tt, no_pi

def random_simulation(g, patterns=10000):
    PI_index = g['forward_index'][(g['forward_level'] == 0) & (g['backward_level'] != 0)]
    no_pi = len(PI_index)
    states = [-1] * len(g['forward_index'])
    full_states = []
    fanin_list = []
    for idx in range(len(g['forward_index'])):
        full_states.append([])
        fanin_list.append([])
    level_list = []
    for level in range(g['forward_level'].max()+1):
        level_list.append([])
    for edge in g['edge_index'].t():
        fanin_list[edge[1].item()].append(edge[0].item())
    for k, idx in enumerate(g['forward_index']):
        level_list[g['forward_level'][k].item()].append(k)
    
    # Simulation 
    for pattern_idx in range(patterns):
        for k, idx in enumerate(PI_index):
            states[idx.item()] = random.randint(0, 1)
        for level in range(1, len(level_list), 1):
            for node_k in level_list[level]:
                source_signals = []
                for pre_k in fanin_list[node_k]:
                    source_signals.append(states[pre_k])
                if len(source_signals) == 0:
                    continue
                states[node_k] = logic(g['gate'][node_k].item(), source_signals)
        for idx in range(len(g['forward_index'])):
            full_states[idx].append(states[idx])
    
    # Incomplete Truth Table / Simulation states
    prob = [0] * len(g['forward_index'])
    for idx in range(len(g['forward_index'])):
        prob[idx] = sum(full_states[idx]) / len(full_states[idx])
    prob = torch.tensor(prob)
    full_states = torch.tensor(full_states)
    return prob, full_states, level_list, fanin_list

def remove_unconnected(x_data, edge_index):
    new_x_data = []
    new_edge_index = []
    is_connected = [False] * len(x_data)
    for edge in edge_index:
        is_connected[edge[0]] = True
        is_connected[edge[1]] = True
    
    new_map = {}
    for idx, is_c in enumerate(is_connected):
        if is_c:
            new_map[idx] = len(new_x_data)
            new_x_data.append(x_data[idx])
    for edge in edge_index:
        new_edge_index.append([new_map[edge[0]], new_map[edge[1]]])
    
    new_x_data = np.array(new_x_data)
    return new_x_data, new_edge_index

def prepare_dg2_labels(graph, no_patterns=10000):
    prob, full_states, level_list, fanin_list = random_simulation(graph, no_patterns)
    # PI Cover
    pi_cover = [[] for _ in range(len(prob))]
    for level in range(len(level_list)):
        for idx in level_list[level]:
            if level == 0:
                pi_cover[idx].append(idx)
            tmp_pi_cover = []
            for pre_k in fanin_list[idx]:
                tmp_pi_cover += pi_cover[pre_k]
            tmp_pi_cover = list(set(tmp_pi_cover))
            pi_cover[idx] += tmp_pi_cover
    # Sample 
    sample_idx = []
    tt_sim_list = []
    
    for node_a in range(0, len(prob)):
        for node_b in range(node_a+1, len(prob)):
            if pi_cover[node_a] != pi_cover[node_b]:
                continue
            if abs(prob[node_a] - prob[node_b]) > 0.1:
                continue
            tt_dis = (full_states[node_a] != full_states[node_b]).sum() / len(full_states[node_a])
            if tt_dis > 0.2 and tt_dis < 0.8:
                continue
            if node_a in fanin_list[node_b] or node_b in fanin_list[node_a]:
                continue
            # if tt_dis == 0 or tt_dis == 1:
            #     continue
            sample_idx.append([node_a, node_b])
            tt_sim_list.append(1-tt_dis)
    
    tt_index = torch.tensor(sample_idx)
    tt_sim = torch.tensor(tt_sim_list)
    return prob, tt_index, tt_sim

def get_sample_paths(g, no_path=1000, max_path_len=128, path_hop_k=0):
    # Parse graph 
    PI_index = g['forward_index'][(g['forward_level'] == 0) & (g['backward_level'] != 0)]
    PO_index = g['forward_index'][(g['forward_level'] != 0) & (g['backward_level'] == 0)]
    no_nodes = len(g['forward_index'])
    level_list = [[] for I in range(g['forward_level'].max()+1)]
    fanin_list = [[] for _ in range(no_nodes)]
    fanout_list = [[] for _ in range(no_nodes)]
    for edge in g['edge_index'].t():
        fanin_list[edge[1].item()].append(edge[0].item())
        fanout_list[edge[0].item()].append(edge[1].item())
    for k, idx in enumerate(g['forward_index']):
        level_list[g['forward_level'][k].item()].append(k)
        
    # Sample Paths
    path_list = []
    path_len_list = []
    path_hash = []
    for _ in range(no_path):
        main_track = []
        path = []
        node_idx = random.choice(PI_index).item()
        path.append(node_idx)
        while len(fanout_list[node_idx]) > 0 and len(path) < max_path_len:
            node_idx = random.choice(fanout_list[node_idx])
            # Add hop
            q = [(node_idx, 0)]
            main_track.append(node_idx)
            while len(q) > 0:
                hop_node_idx, hop_level = q.pop(0)
                if hop_level > path_hop_k:
                    continue
                path.append(hop_node_idx)
                for fanin in fanin_list[hop_node_idx]:
                    q.append((fanin, hop_level+1))
        hash_val = hash_arr(main_track)
        if hash_val in path_hash:
            continue
        else:
            path_hash.append(hash_val)
        path = list(set(path))
        if len(path) < max_path_len:
            path_len_list.append(len(path))
        else:
            path_len_list.append(max_path_len)
        while len(path) < max_path_len:
            path.append(-1)
        path_list.append(path[:max_path_len])
    
    return path_list, path_len_list

def get_fanin_fanout_cone(g, max_no_nodes=512): 
    # Parse graph 
    PI_index = g['forward_index'][(g['forward_level'] == 0) & (g['backward_level'] != 0)]
    PO_index = g['forward_index'][(g['forward_level'] != 0) & (g['backward_level'] == 0)]
    no_nodes = len(g['forward_index'])
    forward_level_list = [[] for I in range(g['forward_level'].max()+1)]
    backward_level_list = [[] for I in range(g['backward_level'].max()+1)]
    fanin_list = [[] for _ in range(no_nodes)]
    fanout_list = [[] for _ in range(no_nodes)]
    for edge in g['edge_index'].t():
        fanin_list[edge[1].item()].append(edge[0].item())
        fanout_list[edge[0].item()].append(edge[1].item())
    for k, idx in enumerate(g['forward_index']):
        forward_level_list[g['forward_level'][k].item()].append(k)
        backward_level_list[g['backward_level'][k].item()].append(k)
    
    # PI Cover 
    pi_cover = [[] for _ in range(no_nodes)]
    for level in range(len(forward_level_list)):
        for idx in forward_level_list[level]:
            if level == 0:
                pi_cover[idx].append(idx)
            tmp_pi_cover = []
            for pre_k in fanin_list[idx]:
                tmp_pi_cover += pi_cover[pre_k]
            tmp_pi_cover = list(set(tmp_pi_cover))
            pi_cover[idx] += tmp_pi_cover
    
    # PO Cover
    po_cover = [[] for _ in range(no_nodes)]
    for level in range(len(backward_level_list)):
        for idx in backward_level_list[level]:
            if level == 0:
                po_cover[idx].append(idx)
            tmp_po_cover = []
            for post_k in fanout_list[idx]:
                tmp_po_cover += po_cover[post_k]
            tmp_po_cover = list(set(tmp_po_cover))
            po_cover[idx] += tmp_po_cover
    
    # fanin and fanout cone 
    fanin_fanout_cones = [[-1]*max_no_nodes for _ in range(max_no_nodes)]
    fanin_fanout_cones = torch.tensor(fanin_fanout_cones, dtype=torch.long)
    for i in range(no_nodes):
        for j in range(no_nodes):
            if i == j:
                fanin_fanout_cones[i][j] = 0
                continue
            if len(pi_cover[j]) <= len(pi_cover[i]) and g['forward_level'][j] < g['forward_level'][i]:
                j_in_i_fanin = True
                for pi in pi_cover[j]:
                    if pi not in pi_cover[i]:
                        j_in_i_fanin = False
                        break
                if j_in_i_fanin:
                    assert fanin_fanout_cones[i][j] == -1
                    fanin_fanout_cones[i][j] = 1
                else:
                    fanin_fanout_cones[i][j] = 0
            elif len(po_cover[j]) <= len(po_cover[i]) and g['forward_level'][j] > g['forward_level'][i]:
                j_in_i_fanout = True
                for po in po_cover[j]:
                    if po not in po_cover[i]:
                        j_in_i_fanout = False
                        break
                if j_in_i_fanout:
                    assert fanin_fanout_cones[i][j] == -1
                    fanin_fanout_cones[i][j] = 2
                else:
                    fanin_fanout_cones[i][j] = 0
            else:
                fanin_fanout_cones[i][j] = 0
    
    assert -1 not in fanin_fanout_cones[:no_nodes, :no_nodes]
    
    return fanin_fanout_cones

def prepare_dg2_labels_cpp(g, no_patterns=15000, 
                           simulator='./src/simulator/simulator', 
                           graph_filepath='./tmp/graph.txt', 
                           res_filepath='./tmp/res.txt'):
    # Parse graph 
    PI_index = g['forward_index'][(g['forward_level'] == 0) & (g['backward_level'] != 0)]
    no_pi = len(PI_index)
    no_nodes = len(g['forward_index'])
    level_list = [[] for I in range(g['forward_level'].max()+1)]
    fanin_list = [[] for _ in range(no_nodes)]
    for edge in g['edge_index'].t():
        fanin_list[edge[1].item()].append(edge[0].item())
    for k, idx in enumerate(g['forward_index']):
        level_list[g['forward_level'][k].item()].append(k)
    
    # PI Cover
    pi_cover = [[] for _ in range(no_nodes)]
    for level in range(len(level_list)):
        for idx in level_list[level]:
            if level == 0:
                pi_cover[idx].append(idx)
            tmp_pi_cover = []
            for pre_k in fanin_list[idx]:
                tmp_pi_cover += pi_cover[pre_k]
            tmp_pi_cover = list(set(tmp_pi_cover))
            pi_cover[idx] += tmp_pi_cover
    
    # Sample TT pairs 
    sample_idx = []
    tt_sim_list = []
    for node_a in range(0, no_nodes):
        for node_b in range(node_a+1, no_nodes):
            if pi_cover[node_a] != pi_cover[node_b]:
                continue
            if node_a in fanin_list[node_b] or node_b in fanin_list[node_a]:
                continue
            sample_idx.append([node_a, node_b])
            tt_sim_list.append([-1])
            
    # Write graph to file
    f = open(graph_filepath, 'w')
    f.write('{} {} {}\n'.format(no_nodes, len(g['edge_index'][0]), no_patterns))
    for idx in range(no_nodes):
        f.write('{} {}\n'.format(g['gate'][idx].item(), g['forward_level'][idx].item()))
    for edge in g['edge_index'].t():
        f.write('{} {}\n'.format(edge[0].item(), edge[1].item()))
    f.write('{}\n'.format(len(sample_idx)))
    for pair in sample_idx:
        f.write('{} {}\n'.format(pair[0], pair[1]))
    f.close()
    
    # Simulation  
    sim_cmd = '{} {} {}'.format(simulator, graph_filepath, res_filepath)
    stdout, exec_time = run_command(sim_cmd)
    f = open(res_filepath, 'r')
    lines = f.readlines()
    f.close()
    prob = [-1] * no_nodes
    for line in lines[:no_nodes]:
        arr = line.replace('\n', '').split(' ')
        prob[int(arr[0])] = float(arr[1])
    for tt_pair_idx, pair in enumerate(sample_idx):
        arr = lines[no_nodes + tt_pair_idx].replace('\n', '').split(' ')
        assert pair[0] == int(arr[0]) and pair[1] == int(arr[1])
        tt_sim_list[tt_pair_idx] = float(arr[2])
    
    tt_index = []
    tt_sim = []
    for pair_idx, pair in enumerate(sample_idx):
        if tt_sim_list[pair_idx] < 0:
            continue
        if tt_sim_list[pair_idx] < 0.2 or tt_sim_list[pair_idx] > 0.8:
            continue
        tt_index.append(pair)
        tt_sim.append(tt_sim_list[pair_idx])
        tt_index.append([pair[1], pair[0]])
        tt_sim.append(tt_sim_list[pair_idx])
    tt_index = torch.tensor(tt_index)
    tt_sim = torch.tensor(tt_sim)
    prob = torch.tensor(prob)
    
    # Remove 
    os.remove(graph_filepath)
    os.remove(res_filepath)
    
    return prob, tt_index, tt_sim

def get_connection_pairs(x_data, edge_index, forward_level, no_src=512, no_dst=512, cone=None):
    no_nodes = len(x_data)
    fanin_list = [[] for _ in range(no_nodes)]
    fanout_list = [[] for _ in range(no_nodes)]
    for edge in edge_index:
        fanin_list[edge[1]].append(edge[0])
        fanout_list[edge[0]].append(edge[1])
    
    # Sample src 
    connect_pair_index = []
    connect_label = []
    for src_k in range(no_src):
        src = random.randint(0, no_nodes-1)
        # Find all connection 
        fanin_cone = []
        fanout_cone = []
        if cone is not None:
            for idx in range(no_nodes):
                if cone[src][idx] == 1:
                    fanin_cone.append(idx)
                elif cone[src][idx] == 2:
                    fanout_cone.append(idx)
        else:
            # Search Fanin cone 
            q = [src]
            while len(q) > 0:
                node_idx = q.pop(0)
                fanin_cone.append(node_idx)
                for pre_k in fanin_list[node_idx]:
                    if pre_k not in fanin_cone:
                        q.append(pre_k)
            # Search Fanout cone 
            q = [src]
            while len(q) > 0:
                node_idx = q.pop(0)
                fanout_cone.append(node_idx)
                for post_k in fanout_list[node_idx]:
                    if post_k not in fanout_cone:
                        q.append(post_k)
        # Sample dst
        dst_k = 0
        while dst_k < no_dst:
            dst_k += 1
            dst = random.randint(0, no_nodes-1)
            while dst == src:
                dst = random.randint(0, no_nodes-1)
            # Check connection 
            pair = [src, dst]
            if pair in connect_pair_index:
                continue
            connect_pair_index.append([src, dst])
            if forward_level[dst] < forward_level[src] and dst in fanin_cone:     # in fanin cone
                connect_label.append(1)
            elif forward_level[dst] > forward_level[src] and dst in fanout_cone:    # in fanout cone
                connect_label.append(2)
            else:
                connect_label.append(0)
    
    connect_pair_index = torch.tensor(connect_pair_index, dtype=torch.long)
    connect_label = torch.tensor(connect_label, dtype=torch.long)
        
    
    return connect_pair_index, connect_label

def get_hop_stru_sim(hop_nodes_list, hop_po_list, edge_index, no_pairs):
    no_hops = len(hop_nodes_list)
    hop_pair_index = np.random.randint(0, no_hops, [no_pairs, 2])
    hop_ged = []
    hop_node_pair = []
    for pair in hop_pair_index:
        if pair[0] == pair[1]:
            ged = 0
        else:
            g1 = nx.DiGraph()
            g2 = nx.DiGraph()
            for edge in edge_index.T:
                if edge[0] in hop_nodes_list[pair[0]] and edge[1] in hop_nodes_list[pair[0]]:
                    g1.add_edge(edge[0].item(), edge[1].item())
                if edge[0] in hop_nodes_list[pair[1]] and edge[1] in hop_nodes_list[pair[1]]:
                    g2.add_edge(edge[0].item(), edge[1].item())
            ged = nx.graph_edit_distance(g1, g2, timeout=0.5)
        
        hop_node_pair.append([hop_po_list[pair[0]], hop_po_list[pair[1]]])
        hop_ged.append(ged)
    hop_node_pair = torch.tensor(hop_node_pair, dtype=torch.long)
    hop_ged = torch.tensor(hop_ged, dtype=torch.float)
    return hop_node_pair, hop_ged
    