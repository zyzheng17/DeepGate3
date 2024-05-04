
import os
import torch
from torch import nn
import time
import random
# from progress.bar import Bar
from torch_geometric.loader import DataLoader
import copy
# from deepgate.arch.mlp import MLP
from deepgate.utils.utils import zero_normalization, AverageMeter, get_function_acc
from deepgate.utils.logger import Logger
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/uac/gds/zyzheng23/projects/DeepGate3-Transformer/src')
from utils.utils import normalize_1
from utils.dag_utils import get_all_hops,get_random_hop
from utils.circuit_utils import complete_simulation, random_simulation

import networkx as nx
from scipy.optimize import linear_sum_assignment

TT_DIFF_RANGE = [0.2, 0.8]

def sample_structural_sim(subgraph, sample_cnt=100):
    stru_sim = []
    candidate_index = list(subgraph.keys())
    init_pair_idx = [random.sample(candidate_index, min(sample_cnt, len(candidate_index))), random.sample(candidate_index, min(sample_cnt, len(candidate_index)))]
    pair_idx = []
    for pair_k in range(sample_cnt):
        g1 = nx.DiGraph()
        g2 = nx.DiGraph()
        graph1 = subgraph[init_pair_idx[0][pair_k]]
        graph2 = subgraph[init_pair_idx[1][pair_k]]
        for edge_idx in range(len(graph1['edges'][0])):
            g1.add_edge(graph1['edges'][0][edge_idx], graph1['edges'][1][edge_idx])
        for edge_idx in range(len(graph2['edges'][0])):
            g2.add_edge(graph2['edges'][0][edge_idx], graph2['edges'][1][edge_idx])
        one_sim = nx.graph_edit_distance(g1, g2, timeout=0.1)
        stru_sim.append(one_sim)
        pair_idx.append([init_pair_idx[0][pair_k], init_pair_idx[1][pair_k]])
    stru_sim = torch.tensor(stru_sim)
    pair_idx = torch.tensor(pair_idx)
    return stru_sim, pair_idx

def sample_functional_tt(subgraph, sample_cnt=100):
    tt_list = []
    no_pi_list = []
    sample_list = []
    candidate_index = list(subgraph.keys())
    sample_list = random.sample(candidate_index, min(sample_cnt, len(candidate_index)))
    for idx in sample_list:
        if idx not in subgraph:
            continue
        g = subgraph[idx]
        tt_bin, no_pi = complete_simulation(g)
        tt_list.append(tt_bin)
        no_pi_list.append(no_pi)
    
    return tt_list, no_pi_list, sample_list

def DeepGate2_Tasks(graph, sample_cnt = 100):
    prob, full_states, level_list, fanin_list = random_simulation(graph, 1024)
    # PI Cover
    pi_cover = [[] for _ in range(len(prob))]
    for level in range(len(level_list)):
        for idx in level_list[level]:
            if level == 0:
                pi_cover[idx].append(idx)
            tmp_pi_cover = []
            for pre_k in fanin_list[idx]:
                tmp_pi_cover += pi_cover[pre_k]
            tmp_pi_cover = list(set(tmp_pi_cover))
            pi_cover[idx] += tmp_pi_cover
    # Sample 
    sample_idx = []
    tt_sim_list = []
    for _ in range(sample_cnt):
        while True:
            node_a = random.randint(0, len(prob)-1)
            node_b = random.randint(0, len(prob)-1)
            if node_a == node_b:
                continue
            if pi_cover[node_a] != pi_cover[node_b]:
                continue
            if abs(prob[node_a] - prob[node_b]) > 0.1:
                continue
            tt_dis = (full_states[node_a] != full_states[node_b]).sum() / len(full_states[node_a])
            if tt_dis > 0.2 and tt_dis < 0.8:
                continue
            if tt_dis == 0 or tt_dis == 1:
                continue
            break
        sample_idx.append([node_a, node_b])
        tt_sim_list.append(1-tt_dis)
    
    tt_index = torch.tensor(sample_idx)
    tt_sim = torch.tensor(tt_sim_list)
    return prob, tt_index, tt_sim

class Trainer():
    def __init__(self, 
                 args, 
                 model, 
                 training_id = 'default',
                 save_dir = './exp', 
                 lr = 1e-4,
                 emb_dim = 128, 
                 device = 'cpu', 
                 batch_size=32, 
                 num_workers=8, 
                 distributed = False, 
                 loss = 'l2',
                 ):
        super(Trainer, self).__init__()
        # Config
        self.args = args
        self.emb_dim = emb_dim
        self.device = device
        self.lr = lr
        self.lr_step = -1
        self.loss_keys = [
            "gate_prob", "gate_lv", "gate_con", "gate_ttsim", 
            "path_onpath", "path_len", "path_and", 
            "hop_tt", "hop_ttsim", "hop_GED", "hop_num", "hop_lv", "hop_onhop"
        ]
        self.loss_weight = {}
        for key in self.loss_keys:
            self.loss_weight[key] = 1.0
        
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.loss = loss
        self.hop_per_circuit = 4
        # Distributed Training 
        self.local_rank = 0
        if self.distributed:
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
            self.device = 'cuda:%d' % self.args.gpus[self.local_rank]
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            print('Training in distributed mode. Device {}, Process {:}, total {:}.'.format(
                self.device, self.rank, self.world_size
            ))
        else:
            print('Training in single device: ', self.device)
            
        if self.local_rank == 0:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.log_dir = os.path.join(save_dir, training_id)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            # Log Path
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log_path = os.path.join(self.log_dir, 'log-{}.txt'.format(time_str))
            
        # Train 
        self.skip_path = False
        self.skip_hop = False
        
        # Loss 
        self.consis_loss_func = nn.MSELoss().to(self.device)
        if self.loss == 'l2':
            self.loss_func = nn.MSELoss().to(self.device)
        elif self.loss == 'l1':
            self.loss_func = nn.L1Loss().to(self.device)
        else:
            raise NotImplementedError
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss().to(self.device)
        self.l1_loss = nn.L1Loss().to(self.device)
        self.ce = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6).to(self.device)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.clf_loss = nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Model
        self.model = model.to(self.device)
        self.model_epoch = 0
        
        # Temp Data 
        self.stru_sim_tmp = {}
        
        # Logger
        if self.local_rank == 0:
            self.logger = Logger(self.log_path)
            
        # Resume 
        if self.args.resume:
            stats = self.resume()
            assert stats
    
    def set_training_args(self, loss_weight={}, lr=-1, lr_step=-1, device='null'):
        if len(loss_weight) > 0:
            for key in loss_weight:
                self.loss_weight[key] = loss_weight[key]
                print('[INFO] Update {} weight from {}'.format(key, loss_weight[key]))
        if lr > 0 and lr != self.lr:
            print('[INFO] Update learning rate from {} to {}'.format(self.lr, lr))
            self.lr = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        if lr_step > 0 and lr_step != self.lr_step:
            print('[INFO] Update learning rate step from {} to {}'.format(self.lr_step, lr_step))
            self.lr_step = lr_step
        if device != 'null' and device != self.device:
            print('[INFO] Update device from {} to {}'.format(self.device, device))
            self.device = device
            self.model = self.model.to(self.device)
            # self.reg_loss = self.reg_loss.to(self.device)
            # self.clf_loss = self.clf_loss.to(self.device)
            self.optimizer = self.optimizer
            # self.readout_rc = self.readout_rc.to(self.device)
        # Check skip 
        tmp_sum = 0
        for key in ["path_onpath", "path_len", "path_and"]:
            tmp_sum += self.loss_weight[key]
        if tmp_sum == 0:
            self.skip_path = True
            print('[INFO] Skip path loss')
        else:
            self.skip_path = False
        tmp_sum = 0
        for key in ["hop_tt", "hop_ttsim", "hop_GED", "hop_num", "hop_lv", "hop_onhop"]:
            tmp_sum += self.loss_weight[key]
        if tmp_sum == 0:
            self.skip_hop = True
            print('[INFO] Skip hop loss')
        else:
            self.skip_hop = False

    def save(self, filename):
        path = os.path.join(self.log_dir, filename)
        data = {
            'epoch': self.model_epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(data, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
        self.model_epoch = checkpoint['epoch']
        self.model.load(path)
        print('[INFO] Continue training from epoch {:}'.format(self.model_epoch))
        return path
    
    def resume(self):
        model_path = os.path.join(self.log_dir, 'model_last.pth')
        if self.local_rank == 0:
            print('[INFO] Load checkpoint from: ', model_path)
        if os.path.exists(model_path):
            self.load(model_path)
            return True
        else:
            return False

    def run_batch(self, batch, phase='train'):

        result_dict = self.model(batch, self.skip_path, self.skip_hop, large_ckt=self.args.enable_cut)


        
        #=========================================================
        #======================GATE-level=========================
        #=========================================================  
             
        # logic probility predction(gate-level)
        l_gate_prob = self.l1_loss(result_dict['node']['prob'], batch.prob.unsqueeze(1).to(self.device))
        # gate level prediction
        l_gate_lv = self.l1_loss(result_dict['node']['level'].squeeze(-1), batch.forward_level.to(self.device))
        #connect classification
        l_gate_con = self.ce(result_dict['node']['connect'],batch.connect_label)
        prob = self.softmax(result_dict['node']['connect'])
        pred_cls = torch.argmax(prob,dim=1)
        con_acc = torch.sum(pred_cls==batch.connect_label) * 1.0 / prob.shape[0]
        #gate pair wise tt sim
        # pred_tt_sim = zero_normalization(result_dict['node']['tt_sim']).to(self.device)
        # tt_sim = zero_normalization(batch.tt_sim).to(self.device)
        # l_gate_ttsim = self.l1_loss(pred_tt_sim, tt_sim)

        # pred_tt_sim = (result_dict['node']['tt_sim']+1)/2
        pred_tt_sim = result_dict['node']['tt_sim'].squeeze()
        l_gate_ttsim = self.l1_loss(pred_tt_sim, batch.tt_sim.to(self.device))
        

        #=========================================================
        #======================PATH-level=========================
        #=========================================================
        if self.args.skip_path:
            l_path_onpath = 0
            on_path_acc = 0
            l_path_len = 0
            l_path_and = 0
        else:
            # on path prediction
            pred_on_path_prob = nn.Sigmoid()(result_dict['path']['on_path']).squeeze(-1).to(self.device)
            l_path_onpath = self.bce(pred_on_path_prob,batch.ninp_labels.float())
            pred_on_path_label = torch.where(pred_on_path_prob>0.5,1,0)
            on_path_acc = (pred_on_path_label==batch.ninp_labels).sum()*1.0/pred_on_path_label.shape[0]

            #path length prediction
            l_path_len = self.l1_loss(result_dict['path']['length'].squeeze(-1), batch.paths_len.to(self.device))

            #path AND&NOT prediction
            l_path_and = self.l1_loss(result_dict['path']['AND'].squeeze(-1), batch.paths_and_ratio.to(self.device))
            # l_path_and = self.l1_loss(result_dict['path']['AND'].squeeze(-1), batch.paths_no_and.to(self.device))
            # l_path_not = self.l1_loss(result_dict['path']['NOT'].squeeze(-1), batch.paths_no_not.to(self.device))

        #=========================================================
        #======================GRAPH-level========================
        #=========================================================

        # Truth table predction(graph-level)
        pred_hop_tt_prob = nn.Sigmoid()(result_dict['hop']['tt']).to(self.device)
        pred_tt = torch.where(pred_hop_tt_prob > 0.5, 1, 0)
        pred_hop_tt_prob = torch.clamp(pred_hop_tt_prob, 1e-6, 1-1e-6)
        l_hop_tt = self.bce(pred_hop_tt_prob, batch.hop_tt.float())
        hamming_dist = torch.mean(torch.abs(pred_tt.float()-batch.hop_tt.float())).cpu()

        # pair-wise tt sim
        # pred_hop_ttsim = zero_normalization(result_dict['hop']['tt_sim']).to(self.device)
        # hop_ttsim = zero_normalization(batch.hop_tt_sim).to(self.device)
        # l_hop_ttsim = self.l1_loss(pred_hop_ttsim, hop_ttsim)

        # pred_hop_ttsim = (result_dict['hop']['tt_sim'].squeeze(-1)+1)/2
        pred_hop_ttsim = result_dict['hop']['tt_sim'].squeeze(-1)
        l_hop_ttsim = self.l1_loss(pred_hop_ttsim, batch.hop_tt_sim.to(self.device))

        #pair wise GED
        # pred_hop_GED = zero_normalization(result_dict['hop']['GED']).to(self.device)
        # hop_GED = zero_normalization(1 - batch.hop_ged).to(self.device)
        # l_hop_GED = self.l1_loss(pred_hop_GED, hop_GED)

        # pred_hop_GED = (result_dict['hop']['GED'].squeeze(-1)+1)/2
        pred_hop_GED = result_dict['hop']['GED'].squeeze(-1)
        l_hop_GED = self.l1_loss(pred_hop_GED, batch.hop_ged.to(self.device))

        #hop num prediction
        l_hop_num = self.l1_loss(result_dict['hop']['area'].squeeze(-1), torch.sum(batch.hop_nodes_stats,dim=1).to(self.device))

        #hop level prediction
        l_hop_lv = self.l1_loss(result_dict['hop']['time'].squeeze(-1), batch.hop_levs.to(self.device))

        #hop on-hop prediction
        pred_on_hop_prob = nn.Sigmoid()(result_dict['hop']['on_hop']).squeeze(-1).to(self.device)
        l_hop_onhop = self.bce(pred_on_hop_prob,batch.ninh_labels.float())
        pred_on_hop_label = torch.where(pred_on_hop_prob>0.5,1,0)
        on_hop_acc = (pred_on_hop_label==batch.ninh_labels).sum()*1.0/pred_on_hop_label.shape[0]


        # Loss 
        sum_weight = 0
        for key in self.loss_weight:
            sum_weight += self.loss_weight[key]
        # loss = l_gate_prob * self.loss_weight['gate_prob'] + \
        #     l_gate_lv * self.loss_weight['gate_lv'] + \
        #     l_gate_con * self.loss_weight['gate_con'] + \
        #     l_gate_ttsim * self.loss_weight['gate_ttsim'] + \
        #     l_path_onpath * self.loss_weight['path_onpath'] + \
        #     l_path_len * self.loss_weight['path_len'] + \
        #     l_path_and * self.loss_weight['path_and'] + \
        #     l_hop_tt * self.loss_weight['hop_tt'] + \
        #     l_hop_ttsim * self.loss_weight['hop_ttsim'] + \
        #     l_hop_GED * self.loss_weight['hop_GED'] + \
        #     l_hop_num * self.loss_weight['hop_num'] + \
        #     l_hop_lv * self.loss_weight['hop_lv'] + \
        #     l_hop_onhop * self.loss_weight['hop_onhop']
        
        func_loss = l_gate_prob * self.loss_weight['gate_prob'] + \
            l_gate_ttsim * self.loss_weight['gate_ttsim'] + \
            l_hop_tt * self.loss_weight['hop_tt'] + \
            l_hop_ttsim * self.loss_weight['hop_ttsim'] 
        
        stru_loss = l_gate_lv * self.loss_weight['gate_lv'] + \
            l_gate_con * self.loss_weight['gate_con'] + \
            l_path_onpath * self.loss_weight['path_onpath'] + \
            l_path_len * self.loss_weight['path_len'] + \
            l_path_and * self.loss_weight['path_and'] + \
            l_hop_GED * self.loss_weight['hop_GED'] + \
            l_hop_num * self.loss_weight['hop_num'] + \
            l_hop_lv * self.loss_weight['hop_lv'] + \
            l_hop_onhop * self.loss_weight['hop_onhop']
        
        # func_loss = 10 * func_loss / sum_weight
        func_loss = func_loss / sum_weight
        stru_loss = stru_loss / sum_weight

        loss = func_loss + stru_loss

        loss_status = {
            'gate_prob': l_gate_prob,
            'gate_lv': l_gate_lv,
            'gate_con': l_gate_con,
            'gate_ttsim': l_gate_ttsim,
            'path_onpath': l_path_onpath,
            'path_len': l_path_len,
            'path_and': l_path_and,
            'hop_tt': l_hop_tt,
            'hop_ttsim': l_hop_ttsim,
            'hop_GED': l_hop_GED,
            'hop_num': l_hop_num,
            'hop_lv': l_hop_lv,
            'hop_onhop': l_hop_onhop,
            'loss' : loss,
            'func_loss':func_loss,
            'stru_loss':stru_loss,
        }

        metric_status = {
            'hamming_dist': hamming_dist,
            'connect_acc': con_acc,
            'on_path_acc': on_path_acc,
            'on_hop_acc': on_hop_acc,
        }
        
        # return loss_status, hamming_dist
        return loss_status, metric_status
    
    def run_dataset(self, epoch, dataset, phase='train'):
        overall_dict = {
            'gate_prob': [],
            'gate_lv': [],
            'gate_con': [],
            'gate_ttsim': [],
            'path_onpath': [],
            'path_len': [],
            'path_and': [],
            'hop_tt': [],
            'hop_ttsim': [],
            'hop_GED': [],
            'hop_num': [],
            'hop_lv': [],
            'hop_onhop': [],
            'loss' : [],
            'func_loss': [],
            'stru_loss': [],
            'hamming_dist': [],
            'connect_acc': [],
            'on_path_acc': [],
            'on_hop_acc': [],
        }
        for iter_id, batch in enumerate(dataset):
            if self.local_rank == 0:
                time_stamp = time.time()
            #TODO：记得改
            # if torch.max(batch.area_idx)>400:
            #     continue

            batch = batch.to(self.device)        
            
            loss_dict, metric_dict = self.run_batch(batch,phase=phase)

            for loss_key in loss_dict:
                if self.args.skip_path and 'path' in loss_key:
                    overall_dict[loss_key].append(torch.tensor(0.))
                else:
                    overall_dict[loss_key].append(loss_dict[loss_key].detach().cpu().item())
            for metric_key in metric_dict:
                if self.args.skip_path and 'path' in metric_key:
                    overall_dict[metric_key].append(torch.tensor(0.))
                else:
                    overall_dict[metric_key].append(metric_dict[metric_key].detach().cpu())

            loss = loss_dict['loss']
            


            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
            
            if self.local_rank == 0:
                # Bar.suffix = '[{:}/{:}] |Tot: {total:} |ETA: {eta:} '.format(iter_id, len(dataset), total=bar.elapsed_td, eta=bar.eta_td)
                # Bar.suffix += '|Prob: {:.4f} |TTCLS: {:.4f} |Loss: {:.4f} |Dist: {:.4f}'.format(
                #     torch.mean(torch.tensor(lprob)).item(), torch.mean(torch.tensor(lttcls)).item(),
                #     torch.mean(torch.tensor(lall)).item(), torch.mean(torch.tensor(hamming_list)).item()
                # )
                # bar.next()
                # bar.suffix = '({phase}) Epoch: {epoch} | Iter: {iter} | Time: {time:.4f}'.format(
                #     phase=phase, epoch=epoch, iter=iter_id, time=time.time()-time_stamp
                # )
                # for loss_key in loss_dict:
                #     if loss_dict[loss_key] !=0:
                #         bar.suffix += ' | {}: {:.4f}'.format(loss_key, loss_dict[loss_key].item())
                # bar.suffix += ' | hamming_dist: {:.4f}'.format(hamming_dist)
                # bar.next()
                output_log = '({phase}) Epoch: {epoch} | Iter: {iter} | Time: {time:.4f} '.format(
                    phase=phase, epoch=epoch, iter=iter_id, time=time.time()-time_stamp
                )
                output_log += '\n======================GATE-level======================== \n'
                gate_loss = 0
                for loss_key in loss_dict:
                    if 'gate' in loss_key:
                        output_log += ' | {}: {:.4f}'.format(loss_key, loss_dict[loss_key].item())
                        gate_loss += loss_dict[loss_key].item()
                output_log += ' | {}: {:.4f}'.format('overall loss', gate_loss)
                
                if not self.args.skip_path:
                    output_log += '\n======================PATH-level======================== \n'
                    path_loss = 0
                    for loss_key in loss_dict:
                        
                        if 'path' in loss_key:
                            output_log += ' | {}: {:.4f}'.format(loss_key, loss_dict[loss_key].item())
                            path_loss+=loss_dict[loss_key].item()
                    output_log += ' | {}: {:.4f}'.format('overall loss', path_loss)
                
                output_log += '\n======================Graph-level======================= \n'
                hop_loss = 0
                for loss_key in loss_dict:
                    if 'hop' in loss_key:
                        
                        output_log += ' | {}: {:.4f}'.format(loss_key, loss_dict[loss_key].item())
                        hop_loss+=loss_dict[loss_key].item()
                output_log += ' | {}: {:.4f}'.format('overall loss', hop_loss)

                output_log += '\n======================All-level========================= \n'
                output_log += ' | {}: {:.4f}'.format('loss', loss_dict['loss'].item())
                output_log += ' | {}: {:.4f}'.format('function loss', loss_dict['func_loss'].item())
                output_log += ' | {}: {:.4f}'.format('structure loss', loss_dict['stru_loss'].item())

                output_log += '\n======================Metric============================\n'
                for metric_key in metric_dict:
                    if metric_dict[metric_key] !=0:
                        output_log += ' | {}: {:.4f}'.format(metric_key, metric_dict[metric_key])
                if iter_id % 5 ==0:
                # if iter_id % 1 ==0:
                    print(output_log)
                    print('\n')

        if self.local_rank == 0:
            for k in overall_dict:
                print('overall {}:{:.4f}'.format(k,torch.mean(torch.tensor(overall_dict[k]))))
            print('\n')

            # output_log = '({phase}) Epoch: {epoch}| '.format(
            #         phase=phase, epoch=epoch
            #     )
            # output_log += '\n======================GATE-level======================== \n'
            # for loss_key in loss_dict:
            #     if 'gate' in loss_key:
            #         output_log += ' | {}: {:.4f}'.format(loss_key, loss_dict[loss_key].item())
            # if not self.args.skip_path:
            #     output_log += '\n======================PATH-level======================== \n'
            #     for loss_key in loss_dict:
            #         if 'path' in loss_key:
            #             output_log += ' | {}: {:.4f}'.format(loss_key, loss_dict[loss_key].item())
            # output_log += '\n======================Graph-level======================= \n'
            # for loss_key in loss_dict:
            #     if 'hop' in loss_key:
            #         output_log += ' | {}: {:.4f}'.format(loss_key, loss_dict[loss_key].item())
            # output_log += '\n======================All-level========================= \n'
            # output_log += ' | {}: {:.4f}'.format('loss', loss_dict['loss'].item())
            # output_log += ' | {}: {:.4f}'.format('function loss', loss_dict['func_loss'].item())
            # output_log += ' | {}: {:.4f}'.format('structure loss', loss_dict['stru_loss'].item())
            # output_log += '\n======================Metric============================\n'

            # for metric_key in metric_dict:
            #     if metric_dict[metric_key] !=0:
            #         output_log += ' | {}: {:.4f}'.format(metric_key, metric_dict[metric_key])
            # # self.logger.write(output_log)
            # # self.logger.write()
            # print(output_log)

    def train(self, num_epoch, train_datasets, val_dataset):

        train_dataset_list = []
        for train_dataset in train_datasets:
            # Distribute Dataset
            if self.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank
                )
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank
                )
                train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                        num_workers=self.num_workers, sampler=train_sampler)
            else:
                train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            train_dataset_list.append(train_dataset)
            
        if self.distributed: 
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                    num_workers=self.num_workers, sampler=val_sampler)
        else:
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        
        
        print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        
        for epoch in range(num_epoch):

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    self.model.to(self.device)
                    for train_dataset in train_dataset_list:
                        self.run_dataset(epoch, train_dataset, phase)
                else:
                    self.model.eval()
                    self.model.to(self.device)
                    # self.run_dataset(epoch, train_dataset_list[-1], phase)
                    self.run_dataset(epoch, val_dataset, phase)

            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                if self.local_rank == 0:
                    print('[INFO] Learning rate decay to {}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            
            # Save model 
            if self.local_rank == 0:
                self.save('model_last.pth')
                if epoch % 10 == 0:
                    self.save('model_{:}.pth'.format(epoch))
                    print('[INFO] Save model to: ', os.path.join(self.log_dir, 'model_{:}.pth'.format(epoch)))
                    
        # del train_dataset
        # del val_dataset
        
    def test(self, val_datasets):

        val_dataset_list = []
        for val_dataset in val_datasets:
            # Distribute Dataset
            if self.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank
                )
                val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                        num_workers=self.num_workers, sampler=sampler)
            else:
                val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            val_dataset_list.append(val_dataset)
        
        self.model.eval()
        self.model.to(self.device)
        for val_dataset in val_dataset_list:
            self.run_dataset(0, val_dataset, 'large_val')

        


