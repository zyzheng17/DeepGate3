# Training script for DeepGate3 by Stone

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np 
from config import get_parse_args

from dg_datasets.dg3_parser import NpzParser
import torch
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

if __name__ == '__main__':
    args = get_parse_args()
    parser = NpzParser(args.data_dir, args.circuit_path, args, random_shuffle=True)
    train_dataset, val_dataset = parser.get_dataset()
    print('Done')
    
    
