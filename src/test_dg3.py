# Training script for DeepGate3 by Stone

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate as dg
import os 
import numpy as np 
import sys
from config import get_parse_args
import time

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
    parser = NpzParser(args.data_dir, args.circuit_path, args, random_shuffle=True, trainval_split=1.0)
    train_dataset, val_dataset = parser.get_dataset()
    
    # Model 
    model = DeepGate3(args)
    model_path = os.path.join('./exp', args.exp_id, 'model_last.pth')
    # model.load(model_path)
    dg2 = dg.Model()
    # dg2.load_pretrained()
    
    # Inference 
    for iter_id, g in enumerate(train_dataset):
        start_time = time.time()
        hs, hf, pred_prob, pred_hop_tt = model(g)
        dg3_runtime = time.time() - start_time
        
        start_time = time.time()
        dg2_hs, dg2_hf = dg2(g)
        dg2_prob = dg2.pred_prob(dg2_hf)
        dg2_runtime = time.time() - start_time
        
        # Prob
        l1 = nn.L1Loss()
        pred_err = l1(pred_prob, g.prob.unsqueeze(1))
        dg2_pred_err = l1(dg2_prob, g.prob.unsqueeze(1))
        
        # TT
        pred_hop_tt_prob = nn.Sigmoid()(pred_hop_tt)
        pred_tt = torch.where(pred_hop_tt_prob > 0.5, 1, 0)
        hamming_dist = torch.mean(torch.abs(pred_tt.float()-g.hop_tt.float()))
        
        # Output 
        print('Circuit: {}'.format(g.name))
        print('Size: {:}, Lev: {:}'.format(len(g.x), g.forward_level.max().item()))
        print('DG3 Time: {:.2f}s, Prob: {:.4f}, TT Dis: {:.4f}'.format(
            dg3_runtime, pred_err.item(), hamming_dist.item()
        ))
        print('DG2 Time: {:.2f}s, Prob: {:.4f}'.format(
            dg2_runtime, dg2_pred_err.item()
        ))
        print()
    
    
