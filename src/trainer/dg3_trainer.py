
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
                 tokenizer,
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
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss().to(self.device)
        self.ce = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6).to(self.device)
        # self.reg_loss = nn.L1Loss().to(self.device)
        # self.clf_loss = nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Model
        self.tokenizer = tokenizer.to(self.device)
        self.model = model.to(self.device)
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
            self.model = self.model.to(self.device)
            # self.reg_loss = self.reg_loss.to(self.device)
            # self.clf_loss = self.clf_loss.to(self.device)
            self.optimizer = self.optimizer
            # self.readout_rc = self.readout_rc.to(self.device)

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

    def run_batch(self, batch):
        
        #initial token emb by dg2
        hs, hf = self.tokenizer(batch)

        #mask graph modeling & Truth table prediction
        logits, tts = self.model(batch, hf)
        #TODO:sigmoid or not
        logits = self.sigmoid(logits)

        loss = self.bce(logits,tts.float())

        #compute hamming distance
        pred_tt = torch.where(logits>0.5,1,0)
        dist = torch.mean(torch.sum(torch.abs(pred_tt - tts),dim=1))

        loss_status = {
            'cls_loss': loss,
            'dist': dist,
        }
        
        return loss_status

    
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
        print(f'save model to {self.log_dir}')
        # self.save(os.path.join(self.log_dir, 'model_-1.pth'))
        self.save('model_last.pth')
        # Train
        print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        for epoch in range(num_epoch): 
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataset = train_dataset
                    self.model.train()
                    self.model.to(self.device)
                else:
                    dataset = val_dataset
                    self.model.eval()
                    self.model.to(self.device)
                    torch.cuda.empty_cache()
                if self.local_rank == 0:
                    bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))
                for iter_id, batch in enumerate(dataset):
                    batch = batch.to(self.device)
                    # emb = self.model(batch)
                    loss = self.run_batch(batch)
                    time_stamp = time.time()
                    
                
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
        
