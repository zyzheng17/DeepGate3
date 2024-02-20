import deepgate as dg 
import numpy as np 
import torch
import os
import copy
import random
from torch_geometric.data import Data, InMemoryDataset
import sys

from deepgate.utils.data_utils import read_npz_file
from typing import Optional, Callable, List
import os.path as osp
sys.path.append('/research/d1/gds/zyzheng23/projects/deepgate3/src')
from utils.dataset_utils import parse_pyg_dg3

from utils.circuit_utils import complete_simulation, prepare_dg2_labels

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        elif key == 'all_hop_pi' or key == 'all_hop_po': 
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index" or key == 'tt_pair_index' or key == 'rc_pair_index':
            return 1
        elif key == 'all_hop_pi' or key == 'all_hop_po' or key == 'all_hop_pi_stats' or key == 'all_tt' or key == 'all_no_hops':
            return 0
        else:
            return 0

class NpzParser():
    def __init__(self, data_dir, circuit_path, args, random_shuffle=True, trainval_split=0.9, debug=False):
        # super().__init__(data_dir, circuit_path, label_path, random_shuffle, trainval_split)
        self.data_dir = data_dir
        dataset = self.inmemory_dataset(data_dir, circuit_path, args, debug=debug)
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
            return osp.join(self.root, name)

        @property
        def raw_file_names(self) -> List[str]:
            return [self.circuit_path]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass

        def process(self):
            data_list = []
            tot_pairs = 0
            circuits = read_npz_file(self.circuit_path)['circuits'].item()
            
            for cir_idx, cir_name in enumerate(circuits):
                print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))

                x_data = circuits[cir_name]['x']
                x_one_hot = dg.construct_node_feature(x_data, 3)
                edge_index = circuits[cir_name]['edge_index']
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                
                if not self.args.enable_large_circuit and x_data.shape[0]>512:
                    continue
                if len(edge_index) == 0:
                    continue

                edge_index = edge_index.t().contiguous()
                forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, x_one_hot.size(0))
                graph = OrderedData()
                graph.x = x_one_hot
                graph.edge_index = edge_index
                graph.name = cir_name
                graph.gate = torch.tensor(x_data[:, 1], dtype=torch.long)
                graph.forward_index = forward_index
                graph.backward_index = backward_index
                graph.forward_level = forward_level
                graph.backward_level = backward_level
                
                # DeepGate2 (node-level) labels
                prob, tt_pair_index, tt_sim = prepare_dg2_labels(graph)
                tt_pair_index = tt_pair_index.t().contiguous()
                graph.prob = prob
                graph.tt_pair_index = tt_pair_index
                graph.tt_sim = tt_sim
                
                # Random select hops 
                rand_idx_list = list(range(len(x_data)))
                random.shuffle(rand_idx_list)
                rand_idx_list = rand_idx_list[0: int(len(x_data) * self.args.hop_ratio)]
                all_hop_pi = torch.zeros((0, 2**(self.args.k_hop-1)), dtype=torch.long)
                all_hop_pi_stats = torch.zeros((0, 2**(self.args.k_hop-1)), dtype=torch.long)
                all_hop_po = torch.zeros((0, 1), dtype=torch.long)
                all_tt = []
                all_no_hops = []
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
                    all_tt.append(hop_tt)
                    all_no_hops.append(no_hops)

                graph.all_hop_pi = all_hop_pi
                graph.all_hop_po = all_hop_po
                graph.all_hop_pi_stats = all_hop_pi_stats
                graph.all_tt = torch.tensor(all_tt, dtype=torch.long)
                graph.all_no_hops = torch.tensor(all_no_hops, dtype=torch.long)
                    
                
                data_list.append(graph)
                
                if self.debug and cir_idx > 10:
                    break

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))
