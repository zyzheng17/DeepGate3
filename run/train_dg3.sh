#!/bin/bash
NUM_PROC=2
GPUS=0,1

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC ./src/train_dg3.py \
 --exp_id train_dg3 \
 --batch_size 64 --lr 0.0001 \
 --en_distrubuted --debug \
 --resume
