import random
import torch
import time
import shlex
import numpy as np 
import subprocess

def run_command(command, timeout=-1):
    try: 
        command_list = shlex.split(command)
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        start_time = time.time()
        while process.poll() is None:
            if timeout > 0 and time.time() - start_time > timeout:
                process.terminate()
                process.wait()
                raise TimeoutError(f"Command '{command}' timed out after {timeout} seconds")

            time.sleep(0.1)

        stdout, stderr = process.communicate()
        stdout = str(stdout).split('\\n')
        return stdout, time.time() - start_time
    except TimeoutError as e:
        return e, -1
    
def hash_arr(arr):
    p = 1543
    md = 6291469
    hash_res = 1
    tmp_arr = arr.copy()
    tmp_arr = np.sort(tmp_arr)
    for ele in tmp_arr:
        hash_res = (hash_res * p + ele) % md
    return hash_res

def normalize_1(data):
    min_val = data.max()
    max_val = data.min()
    normalized_data = (data - min_val) / (max_val - min_val) * 2 - 1
    return normalized_data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def get_function_acc(g, node_emb):
    MIN_GAP = 0.05
    # Sample
    retry = 10000
    tri_sample_idx = 0
    correct = 0
    total = 0
    while tri_sample_idx < 100 and retry > 0:
        retry -= 1
        sample_pair_idx = torch.LongTensor(random.sample(range(len(g.tt_pair_index[0])), 2))
        pair_0 = sample_pair_idx[0]
        pair_1 = sample_pair_idx[1]
        pair_0_gt = g.tt_dis[pair_0]
        pair_1_gt = g.tt_dis[pair_1]
        if pair_0_gt == pair_1_gt:
            continue
        if abs(pair_0_gt - pair_1_gt) < MIN_GAP:
            continue

        total += 1
        tri_sample_idx += 1
        pair_0_sim = torch.cosine_similarity(node_emb[g.tt_pair_index[0][pair_0]].unsqueeze(0), node_emb[g.tt_pair_index[1][pair_0]].unsqueeze(0), eps=1e-8)
        pair_1_sim = torch.cosine_similarity(node_emb[g.tt_pair_index[0][pair_1]].unsqueeze(0), node_emb[g.tt_pair_index[1][pair_1]].unsqueeze(0), eps=1e-8)
        pair_0_predDis = 1 - pair_0_sim
        pair_1_predDis = 1 - pair_1_sim
        succ = False
        if pair_0_gt > pair_1_gt and pair_0_predDis > pair_1_predDis:
            succ = True
        elif pair_0_gt < pair_1_gt and pair_0_predDis < pair_1_predDis:
            succ = True
        if succ:
            correct += 1

    if total > 0:
        acc = correct * 1.0 / total
        return acc
    return -1
            