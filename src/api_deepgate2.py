import deepgate as dg
import os 
import numpy as np 

from models.dg2 import DeepGate2
from utils.dataset_utils import npzitem_to_graph

MODEL_PTH_PATH = './trained/model_last.pth'
npz_path = './data/debug_dg3/LUT6.npz'

if __name__ == '__main__':
    # Create Model 
    model = DeepGate2()
    # Load Model 
    model.load_pretrained(MODEL_PTH_PATH)
    print('Model Loaded from: ', MODEL_PTH_PATH)
    
    # Load dg3 dataset 
    circuits = dg.read_npz_file(npz_path)['circuits'].item()
    for cir_idx, cir_name in enumerate(circuits.keys()):
        x_data = circuits[cir_name]['x_data']
        edge_index = circuits[cir_name]['edge_index']
        tt = circuits[cir_name]['tt']
        graph = npzitem_to_graph(cir_name, x_data, edge_index, tt)
        hs, hf = model(graph)
        print(hs.shape, hf.shape)
    