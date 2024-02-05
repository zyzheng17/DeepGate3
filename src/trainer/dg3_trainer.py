
import os
import torch
from torch import nn
import time
from progress.bar import Bar
from torch_geometric.loader import DataLoader
import copy
# from deepgate.arch.mlp import MLP
from deepgate.utils.utils import zero_normalization, AverageMeter, get_function_acc
from deepgate.utils.logger import Logger
import torch.nn.functional as F
import numpy as np
from utils.utils import normalize_1

import networkx as nx
from scipy.optimize import linear_sum_assignment

TT_DIFF_RANGE = [0.2, 0.8]

class Trainer():
    def __init__(self,
                 dg2,
                 model, 
                 training_id = 'default',
                 save_dir = './exp', 
                 lr = 1e-4,
                 loss_weight = [3.0, 1.0, 2.0],
                 emb_dim = 128, 
                 device = 'cpu', 
                 batch_size=32, 
                 num_workers=8, 
                 distributed = False, 
                 loss = 'l2'
                 ):
        super(Trainer, self).__init__()
        # Config
        self.emb_dim = emb_dim
        self.device = device
        self.lr = lr
        self.lr_step = -1
        self.loss_weight = loss_weight
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.log_dir = os.path.join(save_dir, training_id)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        # Log Path
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.log_path = os.path.join(self.log_dir, 'log-{}.txt'.format(time_str))
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.loss = loss
        
        # Distributed Training 
        self.local_rank = 0
        if self.distributed:
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
            self.device = 'cuda:%d' % self.local_rank
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            print('Training in distributed mode. Device {}, Process {:}, total {:}.'.format(
                self.device, self.rank, self.world_size
            ))
        else:
            print('Training in single device: ', self.device)
        
        # Loss 
        self.consis_loss_func = nn.MSELoss().to(self.device)
        if self.loss == 'l2':
            self.loss_func = nn.MSELoss().to(self.device)
        elif self.loss == 'l1':
            self.loss_func = nn.L1Loss().to(self.device)
        else:
            raise NotImplementedError
        self.bce = nn.BCELoss().to(self.device)
        self.ce = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6).to(self.device)
        # self.reg_loss = nn.L1Loss().to(self.device)
        # self.clf_loss = nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Model
        self.dg2 = dg2.to(self.device)
        self.dg2.eval()
        self.bert = model.to(self.device)
        # self.bert_dec = copy.deepcopy(model).to(self.device)
        self.model_epoch = 0
        
        # Temp Data 
        self.stru_sim_tmp = {}
        
        # Logger
        if self.local_rank == 0:
            self.logger = Logger(self.log_path)
    
    def set_training_args(self, loss_weight=[], lr=-1, lr_step=-1, device='null'):
        if len(loss_weight) == 3 and loss_weight != self.loss_weight:
            print('[INFO] Update loss_weight from {} to {}'.format(self.loss_weight, loss_weight))
            self.loss_weight = loss_weight
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
            self.dg2 = self.dg2.to(self.device)
            self.bert = self.bert.to(self.device)
            # self.reg_loss = self.reg_loss.to(self.device)
            # self.clf_loss = self.clf_loss.to(self.device)
            self.optimizer = self.optimizer
            # self.readout_rc = self.readout_rc.to(self.device)

    def save(self, filename):
        path = os.path.join(self.log_dir, filename)
        data = {
            'epoch': self.model_epoch, 
            'state_dict': self.bert.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(data, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
        self.model_epoch = checkpoint['epoch']
        self.bert.load(path)
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

    # def tt_sim(self,tt,pair_idx):
    #     tt = [bin(int(t,16))[2:] for t in tt]
    #     bs = len(tt)
    #     for i in range(bs):
    #         if len(tt[i])<64:
    #             tt[i] = '0'*(64-len(tt[i]))+tt[i]
        
    #     tt_m = torch.zeros([bs,64])

    #     for i in range(bs):
    #         for j in range(64):
    #             if tt[i][j] == '1':
    #                 tt_m[i,j]=1
    #     tt_m = tt_m.unsqueeze(1).to(self.device)
    #     # tt_s = self.cos_sim(tt_m.repeat(1,bs,1),tt_m.repeat(1,bs,1).permute(1,0,2))
    #     tt_s = self.cos_sim(tt_m[pair_idx[:,0]],tt_m[pair_idx[:,1]])
    #     return tt_s
    
    def tt_sim(self,tt,pair_idx):
        bs = len(tt)
        tt = [bin(int(t,16))[2:].zfill(64) for t in tt]
        tt_m = torch.zeros([bs,64])
        for i in range(bs):
            for j in range(64):
                if tt[i][j] == '1':
                    tt_m[i,j]=1
        tt_s = []
        for k in pair_idx:
            tt_s.append(1 - torch.sum(torch.abs(tt_m[k[0]]-tt_m[k[1]])) * 1.0 / 64)
        tt_s = torch.tensor(tt_s)
        tt_s = tt_s.unsqueeze(1).to(self.device)
        return tt_s
    
    def tt_sim_sample(self,tt,pair_idx):
        bs = len(tt)
        tt = [bin(int(t,16))[2:].zfill(64) for t in tt]
        tt_m = torch.zeros([bs,64])
        for i in range(bs):
            for j in range(64):
                if tt[i][j] == '1':
                    tt_m[i,j]=1
        
        # Sample 
        tt_sim = []
        new_pair_index = []
        for k in pair_idx:
            if k[0] == k[1]:
                continue
            sim = 1 - torch.sum(torch.abs(tt_m[k[0]]-tt_m[k[1]])) * 1.0 / 64
            if sim < TT_DIFF_RANGE[0] or sim > TT_DIFF_RANGE[1]:
                tt_sim.append(sim)
                new_pair_index.append(k)
        tt_sim = torch.tensor(tt_sim)
        tt_sim = tt_sim.unsqueeze(1).to(self.device)
        new_pair_index = torch.tensor(new_pair_index)
        new_pair_index = new_pair_index.to(self.device)
        
        return new_pair_index, tt_sim
    
    def stru_sim_sample(self, batch, pair_idx):
        new_pair_index = pair_idx
        stru_sim = []
        for pair in new_pair_index:
            if pair[0] > pair[1]:
                pair[0], pair[1] = pair[1], pair[0]
            if (pair[0], pair[1]) in self.stru_sim_tmp.keys():
                one_sim = self.stru_sim_tmp[(pair[0], pair[1])]
            else:
                g1 = nx.DiGraph()
                g2 = nx.DiGraph()
                for edge in batch.edge_index.T:
                    if edge[0] >= batch.ptr[pair[0]-1] and edge[0] < batch.ptr[pair[0]]:
                        g1.add_edge(edge[0].item() - batch.ptr[pair[0]-1].item(), edge[1].item() - batch.ptr[pair[0]-1].item())
                        assert edge[1] >= batch.ptr[pair[0]-1] and edge[1] < batch.ptr[pair[0]]
                    if edge[0] >= batch.ptr[pair[1]-1] and edge[0] < batch.ptr[pair[1]]:
                        g2.add_edge(edge[0].item() - batch.ptr[pair[1]-1].item(), edge[1].item() - batch.ptr[pair[1]-1].item())
                        assert edge[1] >= batch.ptr[pair[1]-1] and edge[1] < batch.ptr[pair[1]]
                one_sim = nx.graph_edit_distance(g1, g2, timeout=1)
                self.stru_sim_tmp[(pair[0], pair[1])] = one_sim
            stru_sim.append(one_sim)
            
        stru_sim = torch.tensor(stru_sim).unsqueeze(1).to(self.device)
        return new_pair_index, stru_sim

    def run_batch(self, batch):
        # @ Ziyang: cannot find attribute 'batch_size' in 'batch'
        # bs = batch.batch_size
        bs = len(batch.name)
        # prob = self.dg2.pred_prob(hf)
        graph = batch

        #============function pooling==========================
        segment_info = torch.tensor([0,1,1,1,1,1,1,2])
        segment_info = segment_info.repeat([bs,1]).to(self.device)
        #---------- standard PI-----------
        PI_prob = torch.zeros([graph.x.shape[0],1])
        PI_std = torch.tensor([0.5]*6).unsqueeze(1).repeat(bs,1)
        PI_prob[graph.PIs] = PI_std
        
        hs, hf = self.dg2(batch,PI_prob)
        # p = hf[graph.PIs].detach() # bs*6 128
        std_po = hf[graph.POs].detach() # bs*1 128

        #----------- PI-1 --------------
        PI_prob = torch.zeros([graph.x.shape[0],1])
        PI_1 = torch.rand([6,1]).repeat(bs,1)
        PI_prob[graph.PIs] = PI_1
        
        hs, hf = self.dg2(batch,PI_prob)
        
        pi_emb1 = hf[graph.PIs].detach() # bs*6 128
        po_emb1 = hf[graph.POs].detach() # bs*1 128

        pi_emb1 = pi_emb1.reshape(bs,6,128) # bs 6 128
        po_emb1 = po_emb1.reshape(bs,1,128) # bs 1 128
        
        # Gf_emb1,PO_fhat1 = self.bert(pi_emb1,po_emb1,segment_info)# bs 1 128 ; bs 1 128
        Gf_emb1,PO_fhat1 = self.bert(pi_emb1,std_po,segment_info)# bs 1 128 ; bs 1 128

        #----------- PI-2 -------------
        PI_prob = torch.zeros([graph.x.shape[0],1])
        PI_2 = torch.rand([6,1]).repeat(bs,1)
        PI_prob[graph.PIs] = PI_2
        
        hs, hf = self.dg2(batch,PI_prob)
        
        pi_emb2 = hf[graph.PIs].detach() # 6 128
        po_emb2 = hf[graph.POs].detach() # 1 128

        # @ Ziyang: should be pi_emb2 = pi_emb2 xxx
        # pi_emb2 = pi_emb1.reshape(bs,6,128) # bs 6 128      
        # po_emb2 = po_emb1.reshape(bs,1,128) # bs 1 128
        pi_emb2 = pi_emb2.reshape(bs,6,128) # bs 6 128      
        po_emb2 = po_emb2.reshape(bs,1,128) # bs 1 128
        
        # Gf_emb2,PO_fhat2 = self.bert(pi_emb2,po_emb2,segment_info)
        Gf_emb2,PO_fhat2 = self.bert(pi_emb2,std_po,segment_info)

        #---------------- Function: reconstruction -------------
        l_frec = 0.5*self.loss_func(po_emb1,PO_fhat1) + 0.5*self.loss_func(po_emb2,PO_fhat2) # l_frec = 2.0927

        #--------------- Function: pair-wise similarity -------
        # l_fsim = self.loss_func(emb_sim(Gf_emb1),TT_sim(graph.tt)) + self.loss_func(emb_sim(Gf_emb2),TT_sim(graph.tt))
        pair_idx = np.random.randint(0,bs,[10000,2])
        new_pair_index, tt_s = self.tt_sim_sample(graph.tt, pair_idx)
        if len(new_pair_index) > 0:
            G_sim1 = self.cos_sim(Gf_emb1[new_pair_index[:,0]],Gf_emb1[new_pair_index[:,1]]).reshape(-1)
            G_sim2 = self.cos_sim(Gf_emb2[new_pair_index[:,0]],Gf_emb2[new_pair_index[:,1]]).reshape(-1)

            # @ Ziyang: Cosine similarity or Hamming distance?
            # tt_sim = self.tt_sim(graph.tt,pair_idx).reshape(-1)
            
            # # Normalization
            # G_sim1 = zero_normalization(G_sim1)
            # G_sim2 = zero_normalization(G_sim2)
            # tt_sim = zero_normalization(tt_sim)
            
            # # Softmax 
            # G_sim1 = F.softmax(G_sim1, dim=0)
            # G_sim2 = F.softmax(G_sim2, dim=0)
            # tt_sim = F.softmax(tt_sim, dim=0)
            
            # Normalization -1, 1
            G_sim1 = normalize_1(G_sim1).unsqueeze(1)
            G_sim2 = normalize_1(G_sim2).unsqueeze(1)
            tt_s = normalize_1(tt_s)

            l_fsim = 0.5 * self.loss_func(G_sim1, tt_s) + 0.5 * self.loss_func(G_sim2, tt_s)
        else:
            l_fsim = torch.tensor(0.0).to(self.device)
            
        #----------------- Function: consistency -------------
        l_fcon = self.consis_loss_func(Gf_emb1, Gf_emb2) # 0.5290
        
        #----------------- Structure: pair-wise similarity -------------
        # # 下面的方法求GED Graph Edit Distance 时间复杂度很高 ！！！
        # pair_idx = np.random.randint(0,bs,[1000,2])
        # new_pair_index, G_ged_sim1 = self.stru_sim_sample(batch, pair_idx)

        loss_status = {
            'recon_loss': l_frec, 
            'sim_loss': l_fsim,
            'con_loss': l_fcon
        }
        
        # loss_status = {
        #     'prob_loss': prob_loss, 
        #     'rc_loss': rc_loss,
        #     'func_loss': func_loss
        # }
        
        return hs, hf, loss_status
    
    def train(self, num_epoch, train_dataset, val_dataset):
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
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                     num_workers=self.num_workers, sampler=val_sampler)
        else:
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
        
        # AverageMeter
        batch_time = AverageMeter()
        prob_loss_stats, rc_loss_stats, func_loss_stats = AverageMeter(), AverageMeter(), AverageMeter()
        acc_stats = AverageMeter()
        print(f'save model to {self.log_dir}')
        # self.save(os.path.join(self.log_dir, 'model_-1.pth'))
        self.save('model_last.pth')
        # Train
        print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        for epoch in range(num_epoch): 
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataset = train_dataset
                    self.bert.train()
                    self.bert.to(self.device)
                else:
                    dataset = val_dataset
                    self.bert.eval()
                    self.bert.to(self.device)
                    torch.cuda.empty_cache()
                if self.local_rank == 0:
                    bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))
                for iter_id, batch in enumerate(dataset):
                    batch = batch.to(self.device)
                    time_stamp = time.time()
                    # Get loss
                    hs, hf, loss_status = self.run_batch(batch)

                    loss = loss_status['recon_loss'] * self.loss_weight[0] + \
                        loss_status['sim_loss'] * self.loss_weight[1] + \
                        loss_status['con_loss'] * self.loss_weight[2]
                    loss /= sum(self.loss_weight)
                    loss = loss.mean()
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # Print and save log
                    batch_time.update(time.time() - time_stamp)
                    prob_loss_stats.update(loss_status['recon_loss'].item())
                    rc_loss_stats.update(loss_status['sim_loss'].item())
                    func_loss_stats.update(loss_status['con_loss'].item())

                    # acc = get_function_acc(batch, hf)
                    # acc_stats.update(acc)
                    # if iter_id%10==0:
                    #     print('[{:}/{:}] '.format(iter_id, len(dataset)))
                    #     print('|Recon: {:.4f} |Sim: {:.4f} |Consistency: {:.4f} '.format(prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg))
                    if self.local_rank == 0:
                        Bar.suffix = '[{:}/{:}]|Tot: {total:} |ETA: {eta:} '.format(iter_id, len(dataset), total=bar.elapsed_td, eta=bar.eta_td)
                        Bar.suffix += '|RECS: {:.4f} |SIM: {:.4f} |CONS: {:.4f} '.format(prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg)
                        Bar.suffix += '|Net: {:.2f}s '.format(batch_time.avg)
                        bar.next()
                if phase == 'train' and self.model_epoch % 10 == 0:
                    self.save('model_{:}.pth'.format(self.model_epoch))
                    self.save('model_last.pth')
                if self.local_rank == 0:
                    # self.logger.write('{}| Epoch: {:}/{:} |Prob: {:.4f} |RC: {:.4f} |Func: {:.4f} |ACC: {:.4f} |Net: {:.2f}s\n'.format(
                    #     phase, epoch, num_epoch, prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg, acc_stats.avg, batch_time.avg))
                    # bar.finish()
                    self.logger.write('{}| Epoch: {:}/{:} |Recon: {:.4f} |Sim: {:.4f} |Consistency: {:.4f} |Net: {:.2f}s\n'.format(
                        phase, epoch, num_epoch, prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg, batch_time.avg))
                    bar.finish()
                
                del dataset
            
            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                if self.local_rank == 0:
                    print('[INFO] Learning rate decay to {}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                    
        del train_dataset
        del val_dataset
        
