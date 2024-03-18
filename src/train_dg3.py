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
from models.dg3 import DeepGate3,DeepGate3_structure
from datasets.dataset_utils import npzitem_to_graph
from datasets.dg3_parser import NpzParser
from datasets.aig_parser import AIGParser
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

if __name__ == '__main__':
    args = get_parse_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        
    # Dataset
    # parser = NpzParser(args.data_dir, args.circuit_path, debug=args.debug, random_shuffle=False)
    # parser = NpzParser(args.data_dir, args.circuit_path, args, debug=args.debug, random_shuffle=True)
    parser = NpzParser(args.data_dir, args.circuit_path, args, random_shuffle=True)
    train_dataset, val_dataset = parser.get_dataset()
    
    # Create Model 
    model = DeepGate3(args)
    # model = DeepGate3_structure(args)
    # Train 
    trainer = Trainer(
        args=args, 
        model=model, 
        distributed=args.en_distrubuted, training_id=args.exp_id, batch_size=args.batch_size, device=args.device, 
        loss=args.loss
    )
    trainer.train(args.epoch, train_dataset, val_dataset)
    
    
