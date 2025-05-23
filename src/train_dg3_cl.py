# Training script for DeepGate3 by Stone

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate as dg
import os 
import numpy as np 
import sys
from config import get_parse_args

from models.dg2 import DeepGate2
from models.dg3 import DeepGate3
from dg_datasets.dataset_utils import npzitem_to_graph
from dg_datasets.dg3_parser import NpzParser
from dg_datasets.aig_parser import AIGParser
from dg_datasets.dg3_multi_parser import MultiNpzParser
from bert_model.bert import BERT
from trainer.dg3_trainer import Trainer
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#fix global seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

def get_param(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())  
        Total_params += mulValue  
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue 

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')



if __name__ == '__main__':
    args = get_parse_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        
    # Create Model 
    model = DeepGate3(args)
    # model = DeepGate3_structure(args)
    # Train 
    get_param(model)

    
    # Stone: Support multiple npz files
    parser = MultiNpzParser(args.data_dir, args.npz_dir, args.test_npz_path, args, random_shuffle=True)
    train_dataset, val_dataset = parser.get_dataset()
    
    trainer = Trainer(
        args=args, 
        model=model, 
        distributed=args.en_distrubuted, training_id=args.exp_id, batch_size=args.batch_size, device=args.device, 
        loss=args.loss, 
        num_workers=1
    )
    # # Stone: ICCAD version, no path loss
    
    # Stage 1
    trainer.set_training_args(loss_weight={
        'gate_prob': 2, 
        'gate_lv': 0.5, 
        'gate_con': 1, 
        'gate_ttsim': 0, 
        'hop_tt': 0,
        'hop_ttsim': 0, 
        'hop_GED': 0,
        'hop_num': 0, 
        'hop_lv': 0, 
        'hop_onhop': 0, 
        'path_onpath': 0, 
        'path_len': 0, 
        'path_and': 0
    })
    trainer.train(50, train_dataset, val_dataset)
    
    # Stage 2
    trainer.set_training_args(loss_weight={
        'gate_prob': 2, 
        'gate_lv': 0.3, 
        'gate_con': 1, 
        'gate_ttsim': 1, 
        'hop_tt': 0,
        'hop_ttsim': 0, 
        'hop_GED': 0,
        'hop_num': 0, 
        'hop_lv': 0, 
        'hop_onhop': 0, 
        'path_onpath': 0, 
        'path_len': 0, 
        'path_and': 0
        
    })
    trainer.train(50, train_dataset, val_dataset)
    
    # Stage 3
    trainer.set_training_args(loss_weight={
        'gate_prob': 2, 
        'gate_lv': 0.3, 
        'gate_con': 1, 
        'gate_ttsim': 2, 
        'hop_tt': 1,
        'hop_ttsim': 2, 
        'hop_GED': 2,
        'hop_num': 0.5, 
        'hop_lv': 0.5, 
        'hop_onhop': 1, 
        'path_onpath': 0, 
        'path_len': 0, 
        'path_and': 0
    })
    trainer.train(100, train_dataset, val_dataset)
    
    
