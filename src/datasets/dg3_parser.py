import deepgate as dg 
import numpy as np 
import torch
import os
from torch_geometric.data import Data, InMemoryDataset
import sys

from deepgate.utils.data_utils import read_npz_file
from typing import Optional, Callable, List
import os.path as osp
sys.path.append('/research/d1/gds/zyzheng23/projects/deepgate3/src')
from utils.dataset_utils import parse_pyg_dg3

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
        def __init__(self, root, circuit_path, transform=None, pre_transform=None, pre_filter=None, debug=False):
            self.name = 'npz_inmm_dataset'
            self.root = root
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

                x_data = circuits[cir_name]['x_data']
                edge_index = circuits[cir_name]['edge_index']
                tt = circuits[cir_name]['tt']
                graph = parse_pyg_dg3(x_data, edge_index, tt)
                graph.name = cir_name

                data_list.append(graph)
                
                if self.debug and cir_idx > 10000:
                    break

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))
