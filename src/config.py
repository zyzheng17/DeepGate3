import argparse
import os 
import torch

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='default')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpus', default='0', type=str)
    
    # Dataset
    parser.add_argument('--data_dir', default='./data/train_dg3')
    parser.add_argument('--circuit_path', default='./data/train_dg3/graphs.npz')
    
    # Model 
    parser.add_argument('--pretrained_model_path', default='./trained/model_last.pth')
    parser.add_argument('--dropout', default=0.1, type=float)
    
    # Transformer 
    parser.add_argument('--tf_arch', default='plain', type=str)
    parser.add_argument('--TF_depth', default=4, type=int)
    parser.add_argument('--token_emb', default=128, type=int)
    parser.add_argument('--tf_emb_size', default=128, type=int)
    parser.add_argument('--head_num', default=8, type=int)
    parser.add_argument('--MLP_expansion', default=4, type=int)
    parser.add_argument('--k_hop', default=4, type=float)
    
    # Mask Prediction 
    parser.add_argument('--mlp_hidden', default=128, type=int)
    parser.add_argument('--mlp_layer', default=3, type=int)
    parser.add_argument('--norm_layer', default='batchnorm', type=str)
    parser.add_argument('--act_layer', default='relu', type=str)
    
    # Train
    parser.add_argument('--en_distrubuted', action='store_true')
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--stage2_steps', default=50, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--loss', default='l2', type=str)
    
    # Loss weight
    parser.add_argument('--w_prob', default=1.0, type=float)
    parser.add_argument('--w_tt_sim', default=1.0, type=float)
    parser.add_argument('--w_tt_cls', default=1.0, type=float)
    parser.add_argument('--w_g_sim', default=1.0, type=float)
    
    args = parser.parse_args()
    
    # device
    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]
    args.device = torch.device('cuda:0' if args.gpus[0] >= 0 and torch.cuda.is_available() else 'cpu')
    
    return args