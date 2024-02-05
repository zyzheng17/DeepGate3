#!/bin/bash
NUM_PROC=4
GPUS=0,1,2,3

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC ./src/train_dg2_workload.py \
 --exp_id train_20p \
 --batch_size 64 \
 --data_dir ./data/train_dg2_workload_20p \
 --stage1_steps 0 --stage2_steps 20 \
 --en_distrubuted \
 --resume
