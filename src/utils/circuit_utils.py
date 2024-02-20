import random 
import torch

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

def prepare_dg2_labels(graph):
    prob, full_states, level_list, fanin_list = random_simulation(graph, 1500)
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
    