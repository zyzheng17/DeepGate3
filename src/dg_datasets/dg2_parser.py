import deepgate as dg 
import numpy as np 
import torch
import os
from torch_geometric.data import Data, InMemoryDataset
import sys
from typing import Optional, Callable, List

class NpzParser_Split():
    def __init__(self, data_dir, circuit_path, label_path, \
                 random_shuffle=True, trainval_split=0.9, no_split=10): 
        self.data_dir = data_dir
        dataset = []
        for split_idx in range(no_split):
            dataset.append(self.inmemory_dataset(data_dir, circuit_path, label_path, no_split, split_idx))
            if random_shuffle:
                perm = torch.randperm(len(dataset[-1]))
                dataset[-1] = dataset[-1][perm]
        data_len = len(dataset)
        training_cutoff = int(data_len * trainval_split)
        self.train_dataset = dataset[:training_cutoff]
        self.val_dataset = dataset[training_cutoff:]
    
    def get_dataset(self):
        val_dataset = self.val_dataset[0]
        for dataset in self.val_dataset[1:]:
            val_dataset += dataset
        return self.train_dataset, val_dataset
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, circuit_path, label_path, no_split, split_idx, transform=None, pre_transform=None, pre_filter=None):
            self.name = 'npz_inmm_dataset'
            self.root = root
            self.processed_file_name_list = []
            self.no_split = no_split
            for idx in range(no_split):
                self.processed_file_name_list.append('data_{}.pt'.format(idx))
            self.circuit_path = circuit_path
            self.label_path = label_path
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[split_idx])
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'inmemory'
            return os.path.join(self.root, name)

        @property
        def raw_file_names(self) -> List[str]:
            return [self.circuit_path]

        @property
        def processed_file_names(self) -> str:
            return self.processed_file_name_list

        def download(self):
            pass

        def process(self):
            data_list = []
            tot_pairs = 0
            circuits = dg.read_npz_file(self.circuit_path)['circuits'].item()
            labels = dg.read_npz_file(self.label_path)['labels'].item()
            tot_circuits = len(circuits)
            each_split_circuits = tot_circuits // self.no_split
            no_split = 0
            for cir_idx, cir_name in enumerate(circuits):
                print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
                x = circuits[cir_name]["x"]
                edge_index = circuits[cir_name]["edge_index"]

                tt_dis = labels[cir_name]['tt_dis']
                tt_pair_index = labels[cir_name]['tt_pair_index']
                prob = labels[cir_name]['prob']
                
                rc_pair_index = labels[cir_name]['rc_pair_index']
                is_rc = labels[cir_name]['is_rc']

                if len(tt_pair_index) == 0 or len(rc_pair_index) == 0:
                    print('No tt or rc pairs: ', cir_name)
                    continue

                tot_pairs += len(tt_dis)

                # check the gate types
                # assert (x[:, 1].max() == (len(self.args.gate_to_index)) - 1), 'The gate types are not consistent.'
                graph = dg.parse_pyg_mlpgate(
                    x, edge_index, tt_dis, tt_pair_index, 
                    prob, rc_pair_index, is_rc
                )
                graph.name = cir_name
                data_list.append(graph)
                
                if cir_idx % each_split_circuits == 0 and cir_idx != 0:
                    data, slices = self.collate(data_list)
                    torch.save((data, slices), self.processed_paths[no_split])
                    print('[INFO] Inmemory dataset save: ', self.processed_paths[no_split])
                    print('Total Circuits: {:}'.format(len(data_list)))
                    no_split += 1
                    data_list = []

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[-1])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[-1])
            print('Total Circuits: {:}'.format(len(data_list)))

        def __repr__(self) -> str:
            return f'{self.name}({len(self)})'
