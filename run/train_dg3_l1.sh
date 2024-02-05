#!/bin/bash
NUM_PROC=4
GPUS=0,1,2,3

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC ./src/train_dg3.py \
 --exp_id train_dg3_l1 \
 --batch_size 32 --lr 0.0001 \
 --loss l1 \
 --en_distrubuted --debug 