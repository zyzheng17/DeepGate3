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
# from utils.dataset_utils import parse_pyg_dg3

# from utils.circuit_utils import complete_simulation, prepare_dg2_labels_cpp, \
#     get_fanin_fanout_cone, get_sample_paths, remove_unconnected, \
#     get_connection_pairs, get_hop_pair_labels
    
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

class MultiNpzParser():
    def __init__(self, data_dir, npz_dir, test_npz_path, args, random_shuffle=True):
        self.data_dir = data_dir
        self.npz_dir = npz_dir
        
        # Train Dataset
        self.train_dataset = []
        for npz_id, npz_path in enumerate(os.listdir(npz_dir)):
            print('Parse NPZ: ', npz_path)
            split_dataset = self.inmemory_dataset(data_dir, os.path.join(npz_dir, npz_path), args, npz_id, debug=args.debug)
            if random_shuffle:
                perm = torch.randperm(len(split_dataset))
                split_dataset = split_dataset[perm]
            self.train_dataset.append(split_dataset)
        
        # Test Dataset
        self.test_dataset = self.inmemory_dataset(data_dir, test_npz_path, args, 'test', debug=args.debug)
        
    def get_dataset(self):
        return self.train_dataset, self.test_dataset
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, npz_path, args, npz_id, transform=None, pre_transform=None, pre_filter=None, debug=False):
            self.name = 'npz_inmm_dataset'
            self.npz_id = npz_id
            self.root = root
            self.args = args
            self.npz_path = npz_path
            self.debug = debug
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'immemory_{}'.format(self.npz_id)
            inmemory_path = os.path.join(self.root, name)
            if os.path.exists(inmemory_path):
                print('Inmemory Dataset Path: {}, Existed'.format(inmemory_path))
            else:
                print('Inmemory Dataset Path: {}, New Created'.format(inmemory_path))
            
            return inmemory_path

        @property
        def raw_file_names(self) -> List[str]:
            return [self.npz_path]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass

        def process_npz(self, npz_path):
            data_list = []
            tot_pairs = 0
            circuits = read_npz_file(npz_path)['circuits'].item()
            print('Parse NPZ Datset ...', npz_path)
            tot_time = 0
            
            for cir_idx, cir_name in enumerate(circuits):
                start_time = time.time()
                # print('Parse: {}, {:} / {:} = {:.2f}%, Time: {:.2f}s, ETA: {:.2f}s, Curr Size: {:}'.format(
                #     cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100, 
                #     tot_time, tot_time * (len(circuits) - cir_idx), 
                #     len(data_list)
                # ))
                
                graph = OrderedData()
                succ = True
                for key in circuits[cir_name].keys():
                    if key == 'connect_pair_index' and len(circuits[cir_name][key]) == 0:
                        succ = False
                        break
                    if 'prob' in key or 'sim' in key or 'ratio' in key or 'ged' in key:
                        graph[key] = torch.tensor(circuits[cir_name][key], dtype=torch.float)
                    elif key == 'hs' or key == 'hf':
                        continue
                        graph[key] = torch.tensor(circuits[cir_name][key], dtype=torch.float)
                    else:
                        graph[key] = torch.tensor(circuits[cir_name][key], dtype=torch.long)
                if not succ:
                    continue
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
            self.process_npz(self.npz_path)



class LargeNpzParser():
    def __init__(self, data_dir, npz_dir, test_npz_path, args, random_shuffle=True, trainval_split=0.9):
        self.data_dir = data_dir
        self.npz_dir = npz_dir
        self.train_dataset = []
        for npz_id, npz_path in enumerate(os.listdir(npz_dir)):
            print('Parse NPZ: ', npz_path)
            split_dataset = self.inmemory_dataset(data_dir, os.path.join(npz_dir, npz_path), args, npz_id, debug=args.debug)
            if random_shuffle:
                perm = torch.randperm(len(split_dataset))
                split_dataset = split_dataset[perm]
            self.train_dataset.append(split_dataset)
        dataset = self.train_dataset[0]
        data_len = len(dataset)
        training_cutoff = int(data_len * trainval_split)
        self.train_dataset = [dataset[:training_cutoff]]
        self.val_dataset = dataset[training_cutoff:]

        
    def get_dataset(self):
        return self.train_dataset, self.val_dataset
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, npz_path, args, npz_id, transform=None, pre_transform=None, pre_filter=None, debug=False):
            self.name = 'npz_inmm_dataset'
            self.npz_id = npz_id
            self.root = root
            self.args = args
            self.npz_path = npz_path
            self.debug = debug
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'immemory_{}'.format(self.npz_id)
            inmemory_path = os.path.join(self.root, name)
            if os.path.exists(inmemory_path):
                print('Inmemory Dataset Path: {}, Existed'.format(inmemory_path))
            else:
                print('Inmemory Dataset Path: {}, New Created'.format(inmemory_path))
            
            return inmemory_path

        @property
        def raw_file_names(self) -> List[str]:
            return [self.npz_path]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass

        def process_npz(self, npz_path):
            data_list = []
            tot_pairs = 0
            circuits = read_npz_file(npz_path)['circuits'].item()
            print('Parse NPZ Datset ...', npz_path)
            tot_time = 0
            
            for cir_idx, cir_name in enumerate(circuits):
                start_time = time.time()
                # print('Parse: {}, {:} / {:} = {:.2f}%, Time: {:.2f}s, ETA: {:.2f}s, Curr Size: {:}'.format(
                #     cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100, 
                #     tot_time, tot_time * (len(circuits) - cir_idx), 
                #     len(data_list)
                # ))
                
                graph = OrderedData()
                succ = True
                if circuits[cir_name]['tt_pair_index'].shape[1] == 0 :
                    continue
                
                # if 'area_nodes' not in circuits[cir_name].keys():
                #     print(cir_name)
                #     continue

                for key in circuits[cir_name].keys():
                    if key == 'connect_pair_index' and len(circuits[cir_name][key]) == 0:
                        succ = False
                        break
                    if 'prob' in key or 'sim' in key or 'ratio' in key or 'ged' in key:
                        graph[key] = torch.tensor(circuits[cir_name][key], dtype=torch.float)
                    elif key == 'hs' or key == 'hf':
                        continue
                        graph[key] = torch.tensor(circuits[cir_name][key], dtype=torch.float)

                    else:
                        graph[key] = torch.tensor(circuits[cir_name][key], dtype=torch.long)

                        
                    if key == 'tt_pair_index':
                        max_len = min(circuits[cir_name][key].shape[1], 100000)
                        graph[key] = torch.tensor(circuits[cir_name][key][:,:max_len], dtype=torch.long)
                        graph['tt_sim'] = torch.tensor(circuits[cir_name]['tt_sim'][:max_len], dtype=torch.long)

                    if key == 'connect_pair_index':
                        max_len = min(circuits[cir_name][key].shape[1], 100000)
                        graph[key] = torch.tensor(circuits[cir_name][key][:,:max_len], dtype=torch.long)
                        graph['connect_label'] = torch.tensor(circuits[cir_name]['connect_label'][:max_len], dtype=torch.long)

                    if key == 'hop_pair_index':
                        max_len = min(circuits[cir_name][key].shape[1], 100000)
                        graph[key] = torch.tensor(circuits[cir_name][key][:,:max_len], dtype=torch.long)
                        graph['ninh_labels'] = torch.tensor(circuits[cir_name]['ninh_labels'][:max_len], dtype=torch.long)

                if not succ:
                    continue
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
            self.process_npz(self.npz_path)