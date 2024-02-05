# Training script for DeepGate3 by Stone

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepgate as dg
import os 
import numpy as np 
import sys
import argparse

from models.dg2 import DeepGate2
from datasets.dataset_utils import npzitem_to_graph
from datasets.dg3_parser import NpzParser
from bert_model.bert import BERT
from trainer.dg3_trainer import Trainer
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='default')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    
    # Dataset
    parser.add_argument('--data_dir', default='./data/train_dg3')
    parser.add_argument('--circuit_path', default='./data/train_dg3/graphs.npz')
    
    # Model 
    parser.add_argument('--pretrained_model_path', default='./trained/model_last.pth')
    
    # Train
    parser.add_argument('--en_distrubuted', action='store_true')
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--stage2_steps', default=50, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--loss', default='l2', type=str)
    
    args = parser.parse_args()
    return args

def get_PO(PI,tt):
    return torch.rand_like(PI)

if __name__ == '__main__':

    args = get_parse_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    parser = NpzParser(args.data_dir, args.circuit_path, debug=args.debug, random_shuffle=False)
    train_dataset, val_dataset = parser.get_dataset()

    # Create Model 
    model = DeepGate2()
    # Load Model 
    model.load_pretrained(args.pretrained_model_path)
    print('Model Loaded from: ', args.pretrained_model_path)
    bert_f = BERT(hidden=128)
    # bert_s = BERT(hidden=128)
    # Load dg3 dataset 
    # circuits = dg.read_npz_file(args.circuit_path)['circuits'].item()
    l2loss = nn.MSELoss()
    # dg.Trainer
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        dg2=model, model=bert_f, 
        distributed=args.en_distrubuted, training_id=args.exp_id, batch_size=args.batch_size, device=device, 
        loss=args.loss
    )
    if args.resume:
        assert trainer.resume()
    print('[INFO] Stage 1 Training ...')
    # Stage 1 - Consistency Loss
    trainer.set_training_args(loss_weight=[0, 0, 1.], lr=args.lr)
    trainer.train(30, train_dataset, val_dataset)
    trainer.save('stage1.pth')
    # # Stage 2 - Similarity Loss 
    trainer.set_training_args(loss_weight=[0, 1., 20.], lr=args.lr)
    trainer.train(30, train_dataset, val_dataset)
    trainer.save('stage2.pth')
    # Stage 3 - Full 
    trainer.set_training_args(loss_weight=[1., 1., 10.], lr=args.lr)
    trainer.train(100, train_dataset, val_dataset)
    trainer.save('stage3.pth')

    """
    vocab size:2
        0: mask graph emb
        1: mask PO emb
        
    segment info:
        0: graph
        1: PI
        2: PO
        [graph, PI, PI, PI, PI, PI, PI, PO]
    """