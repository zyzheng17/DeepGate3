import argparse
import os 
import torch

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='default')
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--gpus', default='4', type=str)
    parser.add_argument('--test', action='store_true', default=False)
    
    # Dataset
    parser.add_argument('--data_dir', default='/uac/gds/zyzheng23/projects/DeepGate3-Transformer/data/train_dg3')
    parser.add_argument('--circuit_path', default='/uac/gds/zyzheng23/projects/DeepGate3-Transformer/data/train_dg3/graphs.npz')
    parser.add_argument('--default_dataset', action='store_true')
    parser.add_argument('--hop_ratio', default=0.15, type=float)
    parser.add_argument('--k_hop', default=4, type=int)
    parser.add_argument('--max_hop_pi', default=6, type=int)
    
    # Model 
    parser.add_argument('--pretrained_model_path', default='./DeepGate3-Transformer/trained/model_last.pth')
    parser.add_argument('--dropout', default=0.1, type=float)
    
    # Transformer 
    parser.add_argument('--tf_arch', default='plain', type=str)
    parser.add_argument('--TF_depth', default=4, type=int)
    parser.add_argument('--token_emb', default=128, type=int)
    parser.add_argument('--tf_emb_size', default=128, type=int)
    parser.add_argument('--head_num', default=8, type=int)
    parser.add_argument('--MLP_expansion', default=4, type=int)
    
    # Mask Prediction 
    parser.add_argument('--mlp_hidden', default=128, type=int)
    parser.add_argument('--mlp_layer', default=3, type=int)
    parser.add_argument('--norm_layer', default='batchnorm', type=str)
    parser.add_argument('--act_layer', default='relu', type=str)
    
    # Train
    parser.add_argument('--en_distrubuted', action='store_true')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--stage2_steps', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--loss', default='l2', type=str)
    parser.add_argument('--fast', action='store_true', default=False)
    
    # Loss weight
    parser.add_argument('--w_gate_prob', default=1.0, type=float)
    parser.add_argument('--w_gate_lv', default=1.0, type=float)
    parser.add_argument('--w_gate_con', default=1.0, type=float)
    parser.add_argument('--w_gate_ttsim', default=1.0, type=float)
    parser.add_argument('--w_path_onpath', default=1.0, type=float)
    parser.add_argument('--w_path_len', default=1.0, type=float)
    parser.add_argument('--w_path_and', default=1.0, type=float)
    parser.add_argument('--w_hop_tt', default=1.0, type=float)
    parser.add_argument('--w_hop_ttsim', default=1.0, type=float)
    parser.add_argument('--w_hop_GED', default=1.0, type=float)
    parser.add_argument('--w_hop_num', default=1.0, type=float)
    parser.add_argument('--w_hop_lv', default=1.0, type=float)
    parser.add_argument('--w_hop_onhop', default=1.0, type=float)
    
    args = parser.parse_args()
    
    # device
    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    print('Using GPU:', args.gpus)
    # args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]
    if len(args.gpus) > 1 and torch.cuda.is_available():
        args.en_distrubuted = True
    else:
        args.en_distrubuted = False
    args.device = torch.device('cuda:1' if args.gpus[0] >= 0 and torch.cuda.is_available() else 'cpu')

    # Training 
    args.skip_path = False
    args.skip_hop = False
    
    return args