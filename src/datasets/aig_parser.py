import deepgate as dg 
import numpy as np 
import torch
import glob
import os
from torch_geometric.data import Data, InMemoryDataset
import sys

from deepgate.utils.data_utils import read_npz_file
from typing import Optional, Callable, List
import os.path as osp
from utils.dataset_utils import parse_pyg_dg3

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
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

class AIGParser():
    def __init__(self, data_dir, circuit_path, random_shuffle=True, trainval_split=0.9, debug=False):
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
        def __init__(self, root, circuit_path, transform=None, pre_transform=None, pre_filter=None, debug=False):
            self.name = 'npz_inmm_dataset'
            self.root = root
            self.circuit_path = circuit_path
            self.debug = debug
            self.parser = dg.AigParser()
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
            for aig_path in glob.glob(os.path.join(self.circuit_path, '*.aig')):
                x_data, edge_index = dg.aig_to_xdata(aig_path)
                x_data = np.array(x_data)
                x_one_hot = dg.construct_node_feature(x_data, 3)
                
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                if len(edge_index) == 0:
                    edge_index = edge_index.t().contiguous()
                    forward_index = torch.LongTensor([i for i in range(len(x))])
                    backward_index = torch.LongTensor([i for i in range(len(x))])
                    forward_level = torch.zeros(len(x))
                    backward_level = torch.zeros(len(x))
                else:
                    edge_index = edge_index.t().contiguous()
                    forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, x_one_hot.size(0))

                graph = OrderedData()
                graph.x = x_one_hot
                graph.edge_index = edge_index
                graph.gate = torch.tensor(x_data[:, 1], dtype=torch.long)
                graph.forward_index = forward_index
                graph.backward_index = backward_index
                graph.forward_level = forward_level
                graph.backward_level = backward_level
                graph.name = os.path.basename(aig_path)
                data_list.append(graph)
            
                if self.debug and len(data_list) >= 10:
                    break

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} '.format(len(data_list)))
