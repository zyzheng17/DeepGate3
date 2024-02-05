#!/bin/bash
NUM_PROC=4
GPUS=0,1,2,3

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC ./src/train_dg2_workload.py \
 --exp_id train \
 --batch_size 64 \
 --en_distrubuted \
 --stage1_steps 20 --stage2_steps 20 \
 --sample_ratio 0.1 \
 --resume
