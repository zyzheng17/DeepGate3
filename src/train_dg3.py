# Training script for DeepGate3 by Stone

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate as dg
import os 
import numpy as np 
import sys
from config import get_parse_args

# from models.dg2 import DeepGate2
from models.dg3 import DeepGate3
from dg_datasets.dataset_utils import npzitem_to_graph
from dg_datasets.dg3_parser import NpzParser
from dg_datasets.aig_parser import AIGParser
from dg_datasets.dg3_multi_parser import MultiNpzParser,LargeNpzParser
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
    
    # Stone: Support multiple npz files
    parser = MultiNpzParser(args.data_dir, args.npz_dir, args.test_npz_path, args, random_shuffle=True)
    # parser = LargeNpzParser(args.data_dir, args.npz_dir, args.test_npz_path, args, random_shuffle=False)
    train_dataset, val_dataset = parser.get_dataset()


    # Create Model 
    model = DeepGate3(args)
    if args.dg3_path!=None:
        model.load(args.dg3_path)
    # Train 
    get_param(model)

    # Dataset
    # parser = NpzParser(args.data_dir, args.circuit_path, debug=args.debug, random_shuffle=False)
    # parser = NpzParser(args.data_dir, args.circuit_path, args, debug=args.debug, random_shuffle=True)
    # parser = NpzParser(args.data_dir, args.circuit_path, args, random_shuffle=True)
    # train_dataset, val_dataset = parser.get_dataset()
    # parser = NpzParser(args.data_dir, args.npz_dir, args, random_shuffle=False)

    trainer = Trainer(
        args=args, 
        model=model, 
        distributed=args.en_distrubuted, training_id=args.exp_id, batch_size=args.batch_size, device=args.device, 
        loss=args.loss, 
        num_workers=1
    )

    # # Stone: ICCAD version, no path loss
    loss_keys = [
            "gate_prob", "gate_lv", "gate_con", "gate_ttsim", 
            "path_onpath", "path_len", "path_and", 
            "hop_tt", "hop_ttsim", "hop_GED", "hop_num", "hop_lv", "hop_onhop"
        ]
    
    # trainer.set_training_args(loss_weight={
    #     'gate_prob': 0, 
    #     'gate_lv': 0, 
    #     'gate_con': 0,
    # })
    trainer.set_training_args(loss_weight={
        'path_onpath': 0, 
        'path_len': 0, 
        'path_and': 0,
    })
    # trainer.set_training_args(loss_weight={
    #     'hop_tt': 0, 
    #     'hop_ttsim': 0, 
    #     'hop_GED': 0,
    #     'hop_num': 0,
    #     'hop_lv': 0,
    #     'hop_onhop': 0,
    # })
    trainer.train(args.epoch, train_dataset, val_dataset)
    
    
