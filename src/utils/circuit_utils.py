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

def complete_simulation(g):
    no_pi = len(g['pis'])
    level_list = []
    fanin_list = []
    index_m = {}
    for level in range(g['forward_level'].max()+1):
        level_list.append([])
    for k, idx in enumerate(g['nodes']):
        level_list[g['forward_level'][k].item()].append(k)
        fanin_list.append([])
        index_m[idx.item()] = k
    for edge in g['edges'].t():
        fanin_list[index_m[edge[1].item()]].append(index_m[edge[0].item()])
    
    states = [-1] * len(g['nodes'])
    po_tt = []
    for pattern_idx in range(2**no_pi):
        pattern = [int(x) for x in list(bin(pattern_idx)[2:].zfill(no_pi))]
        for k, idx in enumerate(g['pis']):
            states[index_m[idx.item()]] = pattern[k]
        for level in range(1, len(level_list), 1):
            for node_k in level_list[level]:
                source_signals = []
                for pre_k in fanin_list[node_k]:
                    source_signals.append(states[pre_k])
                if len(source_signals) == 0:
                    continue
                states[node_k] = logic(g['gates'][node_k].item(), source_signals)
        po_tt.append(states[index_m[g['pos'].item()]])
    
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
    
    