import deepgate as dg 
import numpy as np 
import torch
import os
import copy
import random
import time
from torch_geometric.data import Data, InMemoryDataset
import sys

from deepgate.utils.data_utils import read_npz_file
from typing import Optional, Callable, List
import os.path as osp
sys.path.append('/research/d1/gds/zyzheng23/projects/deepgate3/src')
from utils.dataset_utils import parse_pyg_dg3
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
    def __init__(self, data_dir, circuit_path, random_shuffle=True, trainval_split=0.9, debug=False):
        # super().__init__(data_dir, circuit_path, label_path, random_shuffle, trainval_split)
        self.data_dir = data_dir
        dataset = self.inmemory_dataset(data_dir, circuit_path, debug=debug)
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
            tot_time = 0
            
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
                data_list.append(graph)

                if self.debug and cir_idx > 10000:
                    break

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))
