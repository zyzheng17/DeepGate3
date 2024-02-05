import deepgate as dg
import os
import torch
from torch import nn
import time
import random

from progress.bar import Bar
from torch_geometric.loader import DataLoader
from utils.utils import AverageMeter, get_function_acc

class DG2_Trainer(dg.Trainer):
    def __init__(self, model, training_id='default', save_dir='./exp', lr=0.0001, prob_rc_func_weight=..., emb_dim=128, device='cpu', batch_size=32, num_workers=4, distributed=False, \
                 sample_ratio=1.0):
        super().__init__(model, training_id, save_dir, lr, prob_rc_func_weight, emb_dim, device, batch_size, num_workers, distributed)
        self.sample_ratio = sample_ratio
        
    def train(self, num_epoch, train_dataset, val_dataset, skip_val=False):
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
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        
        # AverageMeter
        batch_time = AverageMeter()
        prob_loss_stats, rc_loss_stats, func_loss_stats = AverageMeter(), AverageMeter(), AverageMeter()
        acc_stats = AverageMeter()
        
        # Train
        # print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        for epoch in range(num_epoch): 
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataset = train_dataset
                    self.model.train()
                    self.model.to(self.device)
                else:
                    if skip_val:
                        continue
                    dataset = val_dataset
                    self.model.eval()
                    self.model.to(self.device)
                    torch.cuda.empty_cache()
                if self.local_rank == 0:
                    bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))
                # Random sample batch 
                batch_mask = [0] * len(dataset)
                for batch_idx in range(int(len(batch_mask) * self.sample_ratio)):
                    batch_mask[batch_idx] = 1
                random.shuffle(batch_mask)
                
                for iter_id, batch in enumerate(dataset):
                    if phase == 'train' and self.sample_ratio < 1 and batch_mask[iter_id] == 0:
                        continue
                    batch = batch.to(self.device)
                    time_stamp = time.time()
                    # Get loss
                    hs, hf, loss_status = self.run_batch(batch)
                    loss = loss_status['prob_loss'] * self.prob_rc_func_weight[0] + \
                        loss_status['rc_loss'] * self.prob_rc_func_weight[1] + \
                        loss_status['func_loss'] * self.prob_rc_func_weight[2]
                    loss /= sum(self.prob_rc_func_weight)
                    loss = loss.mean()
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # Print and save log
                    batch_time.update(time.time() - time_stamp)
                    prob_loss_stats.update(loss_status['prob_loss'].item())
                    rc_loss_stats.update(loss_status['rc_loss'].item())
                    func_loss_stats.update(loss_status['func_loss'].item())
                    acc = get_function_acc(batch, hf)
                    acc_stats.update(acc)
                    if self.local_rank == 0:
                        Bar.suffix = '[{:}/{:}]|Tot: {total:} |ETA: {eta:} '.format(iter_id, len(dataset), total=bar.elapsed_td, eta=bar.eta_td)
                        Bar.suffix += '|Prob: {:.4f} |RC: {:.4f} |Func: {:.4f} '.format(prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg)
                        Bar.suffix += '|Acc: {:.2f}%% '.format(acc*100)
                        Bar.suffix += '|Net: {:.2f}s '.format(batch_time.avg)
                        bar.next()
                if phase == 'train' and self.model_epoch % 10 == 0:
                    self.save(os.path.join(self.log_dir, 'model_{:}.pth'.format(self.model_epoch)))
                    self.save(os.path.join(self.log_dir, 'model_last.pth'))
                if self.local_rank == 0:
                    self.logger.write('{}|Epoch: {:}/{:} |Prob: {:.4f} |RC: {:.4f} |Func: {:.4f} |ACC: {:.4f} |Net: {:.2f}s\n'.format(
                        phase, epoch, num_epoch, prob_loss_stats.avg, rc_loss_stats.avg, func_loss_stats.avg, acc_stats.avg, batch_time.avg))
                    bar.finish()
            
            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                if self.local_rank == 0:
                    print('[INFO] Learning rate decay to {}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
        
        # release memory
        del train_dataset
        del val_dataset
        torch.cuda.empty_cache()
        
        