import deepgate as dg
import os 
import numpy as np 
import sys
sys.path.append('/uac/gds/zyzheng23/projects/deepgate3/src')

from dataset_utils import npzitem_to_graph
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader


class dg3_dataset(Dataset):
    def __init__(self,circuit_path):
        super().__init__()
        self.circuits = dg.read_npz_file(circuit_path)['circuits'].item()
        self.cir_names = list(self.circuits.keys())
        self.len = len(self.cir_names)

    def __getitem__(self, index):
        cir_name = self.cir_names[index]
                
        x_data = self.circuits[cir_name]['x_data']
        edge_index = self.circuits[cir_name]['edge_index']
        tt = self.circuits[cir_name]['tt']
        graph = npzitem_to_graph(cir_name, x_data, edge_index, tt)
        # return graph
        return self.circuits[cir_name]

    def __len__(self):
        return self.len

if __name__ == '__main__':
    dataset = dg3_dataset('./deepgate3/dataset/LUT6.npz')
    train_loader = DataLoader(dataset,batch_size=8,shuffle=True,num_workers=8,drop_last=True)
    for data in train_loader:
        print(data)
    dataset[0]