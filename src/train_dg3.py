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

    # Dataset
    # parser = NpzParser(args.data_dir, args.circuit_path, debug=args.debug, random_shuffle=False)
    # parser = NpzParser(args.data_dir, args.circuit_path, args, debug=args.debug, random_shuffle=True)
    parser = NpzParser(args.data_dir, args.circuit_path, args, random_shuffle=True)
    train_dataset, val_dataset = parser.get_dataset()
    

    trainer = Trainer(
        args=args, 
        model=model, 
        distributed=args.en_distrubuted, training_id=args.exp_id, batch_size=args.batch_size, device=args.device, 
        loss=args.loss, 
        num_workers=8
    )
    # # Stone: ICCAD version, no path loss
    # trainer.set_training_args(loss_weight={
    #     'path_onpath': 0, 
    #     'path_len': 0, 
    #     'path_and': 0
    # })
    trainer.train(args.epoch, train_dataset, val_dataset)
    
    
