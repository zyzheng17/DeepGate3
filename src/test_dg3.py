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
from sklearn.metrics import roc_curve, auc

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

def test_tt_sim(g, hf):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tot = 0
    pd_list = []
    gt_list = []

    for pair_index in range(len(g.tt_pair_index[0])):
        pair_A = g.tt_pair_index[0][pair_index]
        pair_B = g.tt_pair_index[1][pair_index]
        pair_gt = g.tt_sim[pair_index]
        pair_pd_sim = torch.cosine_similarity(hf[pair_A].unsqueeze(0), hf[pair_B].unsqueeze(0), eps=1e-8)
        # Skip 
        if abs(pair_gt-0.5) < 0.3:
            continue
        if abs(g.prob[pair_A] - g.prob[pair_B]) > 0.1:
            continue
        
        pd_list.append(pair_pd_sim.item())
        gt_list.append(pair_gt.item() == 0)
        tot += 1
    
    if tot == 0:
        return 0, 0, 0, 0, 0, 0, 0
    
    pd_list = np.array(pd_list)
    gt_list = np.array(gt_list)
    fpr, tpr, thresholds = roc_curve(gt_list, pd_list)
    roc_auc = auc(fpr, tpr)
    opt_thro = thresholds[np.argmax(tpr - fpr)]
    # Threshold
    # pd_list_bin = pd_list > THREHOLD
    pd_list_bin = pd_list > opt_thro

    tp = np.sum(pd_list_bin & gt_list)
    tn = np.sum((~pd_list_bin) & (~gt_list))
    fp = np.sum(pd_list_bin & (~gt_list))
    fn = np.sum((~pd_list_bin) & gt_list)
    
    return tp, tn, fp, fn, tot, roc_auc, opt_thro

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
    model.load(model_path)
    dg2 = dg.Model()
    dg2.load_pretrained()
    
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
        
        # hop TT
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
        
        # Node Pair TT Sim 
        tp, tn, fp, fn, tot, roc_auc, opt_thro = test_tt_sim(g, hf)
        if tot > 0:
            print('DG3 TT Sim: TP: {:.2f}%, TN: {:.2f}%, FP: {:.2f}%, FN: {:.2f}%'.format(
                tp/tot*100, tn/tot*100, fp/tot*100, fn/tot*100
            ))
            print('DG3 ACC: {:.2f}%, Recall: {:.2f}%, Precision: {:.2f}%, F1: {:.2f}'.format(
                (tp+tn)/tot*100, tp/(tp+fn)*100, tp/(tp+fp)*100, 2*tp/(2*tp+fp+fn)
            ))
        tp, tn, fp, fn, tot, roc_auc, opt_thro = test_tt_sim(g, dg2_hf)
        if tot > 0:
            print('DG2 TT Sim: TP: {:.2f}%, TN: {:.2f}%, FP: {:.2f}%, FN: {:.2f}%'.format(
                tp/tot*100, tn/tot*100, fp/tot*100, fn/tot*100
            ))
            print('DG2 ACC: {:.2f}%, Recall: {:.2f}%, Precision: {:.2f}%, F1: {:.2f}'.format(
                (tp+tn)/tot*100, tp/(tp+fn)*100, tp/(tp+fp)*100, 2*tp/(2*tp+fp+fn)
            ))
        
        print()
    
    
