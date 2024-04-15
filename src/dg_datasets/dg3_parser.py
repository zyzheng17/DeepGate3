import deepgate as dg 
import numpy as np 
import torch
import os
import copy
import random
import time
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F

from deepgate.utils.data_utils import read_npz_file
from typing import Optional, Callable, List
from utils.dataset_utils import parse_pyg_dg3

from utils.circuit_utils import complete_simulation, prepare_dg2_labels_cpp, \
    get_fanin_fanout_cone, get_sample_paths, remove_unconnected, \
    get_connection_pairs, get_hop_pair_labels
    
NODE_CONNECT_SAMPLE_RATIO = 0.1
NO_NODE_PATH = 10
NO_NODE_HOP = 10

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):
        # if 'hop_forward_index' in key:
        #     return value.shape[0]
        # elif 'path_forward_index' in key:
        #     return value.shape[0]
        if key == 'ninp_node_index' or key == 'ninh_node_index':
            return self.num_nodes
        elif key == 'ninp_path_index':
            return args[0]['path_forward_index'].shape[0]
        elif key == 'ninh_hop_index':
            return args[0]['hop_forward_index'].shape[0]
        elif key == 'hop_pi' or key == 'hop_po' or key == 'hop_nodes': 
            return self.num_nodes
        elif key == 'winhop_po' or key == 'winhop_nodes':
            return self.num_nodes
        elif key == 'hop_pair_index' or key == 'hop_forward_index':
            return args[0]['hop_forward_index'].shape[0]
        elif key == 'path_forward_index':
            return args[0]['path_forward_index'].shape[0]
        elif key == 'paths' or key == 'hop_nodes':
            return self.num_nodes
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index" or key == 'tt_pair_index' or key == 'rc_pair_index':
            return 1
        elif key == "connect_pair_index" or key == 'hop_pair_index':
            return 1
        elif key == 'hop_pi' or key == 'hop_po' or key == 'hop_pi_stats' or key == 'hop_tt' or key == 'no_hops':
            return 0
        elif key == 'winhop_po' or key == 'winhop_nodes' or key == 'winhop_nodes_stats' or key == 'winhop_lev':
            return 0
        elif key == 'hop_nodes' or key == 'hop_nodes_stats':
            return 0
        elif key == 'paths':
            return 0
        else:
            return 0

class NpzParser():
    def __init__(self, data_dir, circuit_path, args, random_shuffle=True, trainval_split=0.9):
        # super().__init__(data_dir, circuit_path, label_path, random_shuffle, trainval_split)
        self.data_dir = data_dir
        dataset = self.inmemory_dataset(data_dir, circuit_path, args, debug=args.debug)
        if random_shuffle:
            perm = torch.randperm(len(dataset))
            dataset = dataset[perm]
        data_len = len(dataset)
        training_cutoff = int(data_len * trainval_split)
        self.train_dataset = dataset[:training_cutoff]
        self.val_dataset = dataset[training_cutoff:]

    def get_dataset(self):
        return self.train_dataset, self.val_dataset
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, circuit_path, args, transform=None, pre_transform=None, pre_filter=None, debug=False):
            self.name = 'npz_inmm_dataset'
            self.root = root
            self.args = args
            self.circuit_path = circuit_path
            self.debug = debug
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            if self.debug:
                name = 'inmemory_debug'
            else:
                name = 'inmemory'
            if self.args.default_dataset:
                name += '_default'
            inmemory_path = osp.join(self.root, name)
            print('Inmemory Dataset Path: ', inmemory_path)
            return inmemory_path

        @property
        def raw_file_names(self) -> List[str]:
            return [self.circuit_path]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass

        def process_circuits(self):
            data_list = []
            tot_pairs = 0
            circuits = read_npz_file(self.circuit_path)['circuits'].item()
            tot_time = 0
            
            for cir_idx, cir_name in enumerate(circuits):
                start_time = time.time()
                print('Parse: {}, {:} / {:} = {:.2f}%, Time: {:.2f}s, ETA: {:.2f}s, Curr Size: {:}'.format(
                    cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100, 
                    tot_time, tot_time * (len(circuits) - cir_idx), 
                    len(data_list)
                ))

                x_data = circuits[cir_name]['x']
                edge_index = circuits[cir_name]['edge_index']
                x_data, edge_index = remove_unconnected(x_data, edge_index)
                x_one_hot = dg.construct_node_feature(x_data, 3)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                
                if not self.args.default_dataset and x_data.shape[0] > 512:
                    continue
                if len(edge_index) == 0:
                    continue

                edge_index = edge_index.t().contiguous()
                forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, x_one_hot.size(0))
                assert ((forward_level == 0) & (backward_level == 0)).sum() == 0
                
                graph = OrderedData()
                graph.x = x_one_hot
                graph.edge_index = edge_index
                graph.name = cir_name
                graph.gate = torch.tensor(x_data[:, 1], dtype=torch.long).unsqueeze(1)
                graph.forward_index = forward_index
                graph.backward_index = backward_index
                graph.forward_level = forward_level
                graph.backward_level = backward_level
                
                ################################################
                # DeepGate2 (node-level) labels
                ################################################
                prob, tt_pair_index, tt_sim = prepare_dg2_labels_cpp(self.args, graph, 15000)
                assert max(prob).item() <= 1.0 and min(prob).item() >= 0.0
                if len(tt_pair_index) == 0:
                    tt_pair_index = torch.zeros((2, 0), dtype=torch.long)
                else:
                    tt_pair_index = tt_pair_index.t().contiguous()
                graph.prob = prob
                graph.tt_pair_index = tt_pair_index
                graph.tt_sim = tt_sim
                
                if self.args.default_dataset:
                    cone = None
                else:
                    fanin_fanout_cone = get_fanin_fanout_cone(graph)
                    graph.fanin_fanout_cone = fanin_fanout_cone
                    cone = graph.fanin_fanout_cone
                connect_pair_index, connect_label = get_connection_pairs(
                    x_data, edge_index, forward_level, 
                    no_src=int(len(x_data)*NODE_CONNECT_SAMPLE_RATIO), no_dst=int(len(x_data)*NODE_CONNECT_SAMPLE_RATIO),
                    cone=cone
                )
                graph.connect_pair_index = connect_pair_index.T
                graph.connect_label = connect_label
                
                ################################################
                # Path-level labels    
                ################################################             
                sample_paths, sample_paths_len, sample_paths_no_and, sample_paths_no_not = get_sample_paths(graph, no_path=1000, max_path_len=256)
                graph.path_forward_index = torch.tensor(range(len(sample_paths)), dtype=torch.long)
                graph.paths = torch.tensor(sample_paths, dtype=torch.long)
                graph.paths_len = torch.tensor(sample_paths_len, dtype=torch.long)
                graph.paths_and_ratio = torch.tensor(sample_paths_no_and, dtype=torch.long) / torch.tensor(sample_paths_len, dtype=torch.float)
                graph.paths_no_and = torch.tensor(sample_paths_no_and, dtype=torch.long)
                graph.paths_no_not = torch.tensor(sample_paths_no_not, dtype=torch.long)
                # Sample node in path 
                node_path_pair_index = []
                node_path_labels = []
                for path_idx, sample_path in enumerate(sample_paths):
                    path = sample_path[:sample_paths_len[path_idx]]
                    node_in_path = np.random.choice(path, NO_NODE_PATH)
                    node_in_path = [[x, path_idx] for x in node_in_path]
                    node_out_path = [x for x in range(len(x_data)) if x not in path]
                    node_out_path = np.random.choice(node_out_path, NO_NODE_PATH)
                    node_out_path = [[x, path_idx] for x in node_out_path]
                    node_path_pair_index += node_in_path + node_out_path
                    node_path_labels += [1] * NO_NODE_PATH + [0] * NO_NODE_PATH
                node_path_pair_index = torch.tensor(node_path_pair_index, dtype=torch.long)
                ninp_node_index = node_path_pair_index[:, 0]
                ninp_path_index = node_path_pair_index[:, 1]
                graph.ninp_node_index = ninp_node_index
                graph.ninp_path_index = ninp_path_index
                node_path_labels = torch.tensor(node_path_labels, dtype=torch.long)
                graph.ninp_labels = node_path_labels
                
                ################################################
                # Hop-level labels    
                ################################################  
                # Random select hops 
                rand_idx_list = list(range(len(x_data)))
                random.shuffle(rand_idx_list)
                rand_idx_list = rand_idx_list[0: int(len(x_data) * self.args.hop_ratio)]
                all_hop_pi = torch.zeros((0, 2**(self.args.k_hop-1)), dtype=torch.long)
                all_hop_pi_stats = torch.zeros((0, 2**(self.args.k_hop-1)), dtype=torch.long)
                all_hop_po = torch.zeros((0, 1), dtype=torch.long)
                max_hop_nodes_cnt = 0
                for k in range(self.args.k_hop+1):
                    max_hop_nodes_cnt += 2**k
                all_hop_nodes = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
                all_hop_nodes_stats = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
                all_tt = []
                all_hop_nodes_cnt = []
                all_hop_level_cnt = []
                for idx in rand_idx_list:
                    last_target_idx = copy.deepcopy([idx])
                    curr_target_idx = []
                    hop_nodes = []
                    hop_edges = torch.zeros((2, 0), dtype=torch.long)
                    for k_hops in range(self.args.k_hop):
                        if len(last_target_idx) == 0:
                            break
                        for n in last_target_idx:
                            ne_mask = edge_index[1] == n
                            curr_target_idx += edge_index[0, ne_mask].tolist()
                            hop_edges = torch.cat([hop_edges, edge_index[:, ne_mask]], dim=-1)
                            hop_nodes += edge_index[0, ne_mask].unique().tolist()
                        last_target_idx = list(set(curr_target_idx))
                        curr_target_idx = []

                    if len(hop_nodes) < 2:
                        continue
                    hop_nodes = torch.tensor(hop_nodes).unique().long()
                    hop_nodes = torch.cat([hop_nodes, torch.tensor([idx])])
                    no_hops = k_hops + 1
                    hop_forward_level, hop_forward_index, hop_backward_level, _ = dg.return_order_info(hop_edges, len(x_data))
                    hop_forward_level = hop_forward_level[hop_nodes]
                    hop_backward_level = hop_backward_level[hop_nodes]
                    
                    hop_gates = graph.gate[hop_nodes]
                    hop_pis = hop_nodes[(hop_forward_level==0) & (hop_backward_level!=0)]
                    hop_pos = hop_nodes[(hop_forward_level!=0) & (hop_backward_level==0)]
                    if len(hop_pis) > 2**(self.args.k_hop-1):
                        continue
                    
                    hop_pi_stats = [2] * len(hop_pis)  # -1 Padding, 0 Logic-0, 1 Logic-1, 2 variable
                    for assigned_pi_k in range(self.args.max_hop_pi, len(hop_pi_stats), 1):
                        hop_pi_stats[assigned_pi_k] = random.randint(0, 1)
                    hop_tt, _ = complete_simulation(hop_pis, hop_pos, hop_forward_level, hop_nodes, hop_edges, hop_gates, pi_stats=hop_pi_stats)
                    while len(hop_tt) < 2**self.args.max_hop_pi:
                        hop_tt += hop_tt
                        hop_pis = torch.cat([torch.tensor([-1]), hop_pis])
                        hop_pi_stats.insert(0, -1)
                    while len(hop_pi_stats) < 2**(self.args.k_hop-1):
                        hop_pis = torch.cat([torch.tensor([-1]), hop_pis])
                        hop_pi_stats.insert(0, -1)
                    
                    # Record the hop 
                    all_hop_pi = torch.cat([all_hop_pi, hop_pis.view(1, -1)], dim=0)
                    all_hop_po = torch.cat([all_hop_po, hop_pos.view(1, -1)], dim=0)
                    all_hop_pi_stats = torch.cat([all_hop_pi_stats, torch.tensor(hop_pi_stats).view(1, -1)], dim=0)
                    assert len(hop_nodes) <= max_hop_nodes_cnt
                    all_hop_nodes_cnt.append(len(hop_nodes))
                    all_hop_level_cnt.append(no_hops)
                    hop_nodes_stats = torch.ones(len(hop_nodes), dtype=torch.long)
                    hop_nodes = F.pad(hop_nodes, (0, max_hop_nodes_cnt - len(hop_nodes)), value=-1)
                    hop_nodes_stats = F.pad(hop_nodes_stats, (0, max_hop_nodes_cnt - len(hop_nodes_stats)), value=0)
                    all_hop_nodes = torch.cat([all_hop_nodes, hop_nodes.view(1, -1)], dim=0)
                    all_hop_nodes_stats = torch.cat([all_hop_nodes_stats, hop_nodes_stats.view(1, -1)], dim=0)
                    all_tt.append(hop_tt)

                graph.hop_pi = all_hop_pi
                graph.hop_po = all_hop_po
                graph.hop_pi_stats = all_hop_pi_stats
                graph.hop_nodes = all_hop_nodes
                graph.hop_nodes_stats = all_hop_nodes_stats
                graph.hop_tt = torch.tensor(all_tt, dtype=torch.long)
                graph.hop_nds = torch.tensor(all_hop_nodes_cnt, dtype=torch.long)
                graph.hop_levs = torch.tensor(all_hop_level_cnt, dtype=torch.long)
                graph.hop_forward_index = torch.tensor(range(len(all_hop_nodes)), dtype=torch.long)
                
                hop_pair_index, hop_pair_ged, hop_pair_tt_sim = get_hop_pair_labels(all_hop_nodes, graph.hop_tt, edge_index, no_pairs=int(len(all_hop_nodes) * 0.1))
                no_pairs = len(hop_pair_index)
                if no_pairs == 0:
                    continue
                graph.hop_pair_index = hop_pair_index.T.reshape(2, no_pairs)
                graph.hop_ged = hop_pair_ged
                graph.hop_tt_sim = torch.tensor(hop_pair_tt_sim, dtype=torch.float)
                
                # Sample node in hop 
                node_hop_pair_index = []
                node_hop_labels = []
                for hop_idx, sample_hop in enumerate(all_hop_nodes):
                    hop = sample_hop[sample_hop != -1].tolist()
                    node_in_hop = np.random.choice(hop, NO_NODE_HOP)
                    node_in_hop = [[x, hop_idx] for x in node_in_hop]
                    node_out_hop = [x for x in range(len(x_data)) if x not in hop]
                    node_out_hop = np.random.choice(node_out_hop, NO_NODE_HOP)
                    node_out_hop = [[x, hop_idx] for x in node_out_hop]
                    node_hop_pair_index += node_in_hop + node_out_hop
                    node_hop_labels += [1] * NO_NODE_HOP + [0] * NO_NODE_HOP
                node_hop_pair_index = torch.tensor(node_hop_pair_index, dtype=torch.long)
                node_hop_labels = torch.tensor(node_hop_labels, dtype=torch.long)
                ninh_node_index = node_hop_pair_index[:, 0]
                ninh_hop_index = node_hop_pair_index[:, 1]
                graph.ninh_node_index = ninh_node_index
                graph.ninh_hop_index = ninh_hop_index
                graph.ninh_labels = node_hop_labels
                
                # Statistics
                graph.no_nodes = len(x_data)
                graph.no_edges = len(edge_index[0])
                graph.no_hops = len(all_hop_nodes)
                graph.no_paths = len(sample_paths)
                data_list.append(graph)
                tot_time = time.time() - start_time
                
                if self.debug and cir_idx > 40:
                    break

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))
        
        def process_npz(self, npz_path):
            data_list = []
            tot_pairs = 0
            circuits = read_npz_file(npz_path)['circuits'].item()
            tot_time = 0
            
            for cir_idx, cir_name in enumerate(circuits):
                start_time = time.time()
                print('Parse: {}, {:} / {:} = {:.2f}%, Time: {:.2f}s, ETA: {:.2f}s, Curr Size: {:}'.format(
                    cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100, 
                    tot_time, tot_time * (len(circuits) - cir_idx), 
                    len(data_list)
                ))
                
                graph = OrderedData()
                for key in circuits[cir_name].keys():
                    if 'prob' in key or 'sim' in key or 'ratio' in key or 'ged' in key:
                        graph[key] = torch.tensor(circuits[cir_name][key], dtype=torch.float)
                    else:
                        graph[key] = torch.tensor(circuits[cir_name][key], dtype=torch.long)
                graph.name = cir_name
                data_list.append(graph)
                tot_time = time.time() - start_time
                
                if self.debug and cir_idx > 40:
                    break
                
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))
                
        def process(self):
            if self.args.load_npz == '':
                self.process_circuits()
            else:
                self.process_npz(self.args.load_npz)