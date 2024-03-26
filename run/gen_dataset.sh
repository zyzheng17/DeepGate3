#!/bin/bash
nohup python3 -u ./src/gen_dataset.py \
 --exp_id default \
 --gpus -1 \
 --data_dir ./data/train_dg3 \
 --circuit_path ./data/train_dg3/graphs.npz >> /uac/gds/zyzheng23/projects/DeepGate3-Transformer/data/data.log 2>&1 &
