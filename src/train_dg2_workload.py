import deepgate 
import os 
import numpy as np 
import argparse
import torch

from models.dg2 import DeepGate2
from trainers.dg2_trainer import DG2_Trainer
from datasets.dg2_parser import NpzParser_Split

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='default')
    parser.add_argument('--local_rank', default=0, type=int)
    
    # Dataset
    parser.add_argument('--data_dir', default='./data/train_dg2_workload')
    parser.add_argument('--circuit_path', default='../DeepGate3_Dataset/deepgate3_dataset/graphs.npz')
    parser.add_argument('--label_path', default='../DeepGate3_Dataset/deepgate3_dataset/labels.npz')
    
    # Train
    parser.add_argument('--en_distrubuted', action='store_true')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stage1_steps', default=20, type=int)
    parser.add_argument('--stage2_steps', default=50, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float)
    
    # Dataset 
    parser.add_argument('--no_split', default=10, type=int)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parse_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    # parser = deepgate.NpzParser(args.data_dir, args.circuit_path, args.label_path)
    parser = NpzParser_Split(args.data_dir, args.circuit_path, args.label_path, no_split=args.no_split)
    train_dataset, val_dataset = parser.get_dataset()
    
    # Create Model 
    model = DeepGate2()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = DG2_Trainer(
        model, training_id=args.exp_id, batch_size=args.batch_size, 
        device=device, distributed=args.en_distrubuted, num_workers=0, 
        sample_ratio=args.sample_ratio
    )
    if args.resume:
        assert trainer.resume()
    print('[INFO] Stage 1 Training ...')
    trainer.set_training_args(prob_rc_func_weight=[3.0, 1.0, 0.0], lr=1e-4)
    for stage_step_idx in range(args.stage1_steps):
        for split_idx in range(args.no_split):
            print('# Epoch: {} / # Split: {}'.format(stage_step_idx, split_idx))
            skip_val = False if split_idx == args.no_split - 1 else True
            trainer.train(1, train_dataset[split_idx], val_dataset, skip_val=skip_val)
    
    print('[INFO] Stage 2 Training ...')
    trainer.set_training_args(prob_rc_func_weight=[3.0, 1.0, 2.0], lr=1e-4)
    for stage_step_idx in range(args.stage2_steps):
        for split_idx in range(args.no_split):
            print('# Epoch: {} / # Split: {}'.format(stage_step_idx, split_idx))
            skip_val = False if split_idx == args.no_split - 1 else True
            trainer.train(1, train_dataset[split_idx], val_dataset, skip_val=skip_val)
    

